from datetime import timedelta
import os
import sys
import types
import asyncio
import inspect
import modal
import dataclasses
from functools import wraps
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from typing import (
    Coroutine,
    Any,
    Callable,
    ParamSpec,
    TypeVar,
    Union,
    Sequence,
    cast,
    dataclass_transform,
)
from modal.partial_function import PartialFunction as _ModalPartialFunction
import threading
import temporalio.common
import temporalio.converter
from temporalio import activity
from temporalio.activity import Info, _Context, _ActivityCancellationDetailsHolder
from temporalio.exceptions import ApplicationError
from temporalio.worker import (
    Worker,
    Interceptor,
    ActivityInboundInterceptor,
    ExecuteActivityInput,
)
from temporalio.client import Client, AsyncActivityHandle
from async_lru import alru_cache

P = ParamSpec("P")
R = TypeVar("R")
T = TypeVar("T")

# Heartbeat hand-off between the dispatcher and the Modal worker. While an
# activity waits on Modal's input queue the dispatcher heartbeats on its behalf;
# once a worker starts it writes the "started" marker into the shared modal.Dict
# to take over, and removes the key entirely when the activity finishes.
_HEARTBEAT_COORD_DICT_NAME = "modal-temporal-heartbeat-coord"


def _coord_key(info: Info) -> str:
    """Coordination key for an activity's heartbeat hand-off.

    Per-attempt: a preempted attempt's finally-cleanup (coord_dict.pop) runs
    during Modal's SIGTERM grace and would otherwise clobber the next attempt's
    queued marker under a stable key, leaving the retry's cold start
    un-heartbeated. Costs one orphaned key per crashed attempt (modal.Dict has
    no TTL)."""
    return f"{info.workflow_id}/{info.workflow_run_id}/{info.activity_id}/{info.attempt}"


def _queued_marker(info: Info) -> str:
    # Attempt-tagged so a slow-to-die zombie worker from a prior attempt (which
    # can only write started:<old>) can never move the current attempt's loop.
    return f"queued:{info.attempt}"


def _started_marker(info: Info) -> str:
    return f"started:{info.attempt}"

# Keep references to in-flight dispatcher heartbeat loops so they are not
# garbage-collected while running (asyncio only holds weak refs to tasks).
_dispatcher_heartbeat_tasks: set[asyncio.Task] = set()


@alru_cache(maxsize=1)
async def get_heartbeat_coord_dict() -> modal.Dict:
    return modal.Dict.from_name(_HEARTBEAT_COORD_DICT_NAME, create_if_missing=True)


async def _auto_heartbeat_loop(
    handle: AsyncActivityHandle,
    activity_name: str,
    heartbeat_timeout: timedelta,
    activity_task: asyncio.Task,
) -> None:
    """Temporal keeps a heartbeat to make sure the activity is running."""
    interval = heartbeat_timeout.total_seconds() / 2.0
    while True:
        try:
            # Heartbeat the instant the worker takes over from the dispatcher,
            # before the first interval elapses.
            await handle.heartbeat()
            print(f"[external worker] heartbeat sent for {activity_name}")
            await asyncio.sleep(interval)
        except asyncio.CancelledError:
            return
        except Exception as e:
            print(f"[external worker] heartbeat failed: {e}")
            # Heartbeat failure usually means Temporal cancelled or expired the activity.
            activity_task.cancel()
            return


async def _dispatcher_heartbeat_loop(
    handle: AsyncActivityHandle,
    info: Info,
    coord_dict: modal.Dict,
) -> None:
    """Heartbeat on behalf of an activity while it waits on Modal's input queue.

    The dispatcher owns the heartbeat from the moment the activity is spawned
    until a Modal worker picks it up. The worker signals takeover by writing the
    started:<attempt> marker into the shared modal.Dict; completion removes the
    key entirely. In either case this loop exits and the worker takes over."""
    assert info.heartbeat_timeout is not None
    key = _coord_key(info)
    queued = _queued_marker(info)
    interval = info.heartbeat_timeout.total_seconds() / 2.0
    while True:
        # On a retry the dispatcher is the sole heartbeater while the container
        # cold-starts, so heartbeat before the first sleep.
        # Step down once our marker is gone: the worker took over
        # (started:<attempt>), the activity finished (key removed), or a newer
        # attempt superseded us (queued:<attempt+1>).
        if await coord_dict.get.aio(key) != queued:
            return
        try:
            await handle.heartbeat()
            print(f"[dispatcher] heartbeat (queued) for {info.activity_type}")
        except Exception as e:
            print(f"[dispatcher] queued heartbeat failed for {info.activity_type}: {e}")
            return
        await asyncio.sleep(interval)


async def run_activity(fn: Callable, args: Any, client: Client, info: Info):
    loop = asyncio.get_running_loop()
    handle = client.get_async_activity_handle(task_token=info.task_token)

    def heartbeat_fn(*details: Any) -> None:
        asyncio.run_coroutine_threadsafe(handle.heartbeat(*details), loop)

    # This uses temporal private API to construct the _Context. If you do not need the context
    # i.e. call `activity.heartbeat` then it's okay to not set the context.
    context = _Context(
        info=lambda: info,
        heartbeat=heartbeat_fn,
        cancelled_event=temporalio.common._CompositeEvent(
            thread_event=threading.Event(), async_event=asyncio.Event()
        ),
        worker_shutdown_event=temporalio.common._CompositeEvent(
            thread_event=threading.Event(), async_event=asyncio.Event()
        ),
        shield_thread_cancel_exception=None,
        payload_converter_class_or_instance=temporalio.converter.DefaultPayloadConverter,
        runtime_metric_meter=None,
        client=client,
        cancellation_details=_ActivityCancellationDetailsHolder(),
    )
    token = _Context.set(context)
    try:
        if inspect.iscoroutinefunction(fn):
            coro = fn(*args)
        else:
            coro = asyncio.to_thread(fn, *args)
        return await run_activity_with_temporal(coro, fn.__name__, handle, info)
    finally:
        _Context.reset(token)


async def run_activity_with_temporal(
    coro: Coroutine,
    activity_name: str,
    handle: AsyncActivityHandle,
    info: Info,
):
    activity_task = asyncio.create_task(coro)
    heartbeat_task = None
    coord_dict = None
    if info.heartbeat_timeout:
        # Take over heartbeat duty from the dispatcher: mark the activity as
        # started so the dispatcher's queue-side heartbeat loop steps down.
        coord_dict = await get_heartbeat_coord_dict()
        await coord_dict.put.aio(_coord_key(info), _started_marker(info))
        heartbeat_task = asyncio.create_task(
            _auto_heartbeat_loop(
                handle, activity_name, info.heartbeat_timeout, activity_task
            )
        )

    try:
        result = await activity_task
        await handle.complete(result)
        print(f"[external worker] completed {activity_name} -> {result!r}")
        return result
    except asyncio.CancelledError:
        await handle.report_cancellation()
        return
    except Exception as e:
        await handle.fail(ApplicationError(str(e)))
        print(f"[external worker] failed {activity_name}: {e}")
    finally:
        if heartbeat_task:
            heartbeat_task.cancel()
        if coord_dict is not None:
            # Remove the coordination key so the dispatcher loop (if any is
            # still alive) sees the activity is done and exits.
            await coord_dict.pop.aio(_coord_key(info), None)


@alru_cache(maxsize=1)
async def get_temporal_client() -> Client:
    """Get temporal client.

    You may need to pass in additional `secrets=` to `@modal_activity` and ingest it here to
    authenticate with Temporal."""
    return await Client.connect(
        os.environ["TEMPORAL_SERVER_URL"],
        api_key=os.getenv("TEMPORAL_API_KEY"),
        namespace=os.getenv("TEMPORAL_NAMESPACE", "default"),
    )


@dataclass(frozen=True)
class _Runner:
    modal_name: str  # name the handler is deployed under in the Modal app
    is_class: bool


# Temporal activity_type -> how the dispatcher reaches its Modal handler.
# Populated at import time, read by DispatchActivityInterceptor. There is no
# naming convention: each activity records its handler here explicitly.
REGISTRY: dict[str, _Runner] = {}


def modal_activity(
    app: modal.App, **modal_opts: Any
) -> Callable[[Union[Callable[P, R], _ModalPartialFunction[P, R, R]]], Callable[P, R]]:
    """Register a function as a Temporal activity AND build + record the Modal
    function that runs it. Returns the Temporal activity to pass to the Worker."""

    def decorate(
        f: Union[Callable[P, R], _ModalPartialFunction[P, R, R]],
    ) -> Callable[P, R]:
        # Unwrap Modal PartialFunction (e.g. from @modal.concurrent) to get
        # the plain callable that temporalio's activity.defn requires.
        partial = None
        inner = None
        if isinstance(f, _ModalPartialFunction):
            inner = getattr(f, getattr(f, "_sync_synchronizer")._original_attr)
            partial = f
            raw_f = inner.raw_f
        else:
            raw_f = f

        temporal_activity: Callable[P, R] = activity.defn(raw_f)

        @wraps(raw_f)
        async def runner(info: Info, /, args: Any):
            client = await get_temporal_client()
            return await run_activity(raw_f, args, client, info)

        modal_name = f"{raw_f.__name__}_runner"
        runner.__name__ = runner.__qualname__ = modal_name
        # Modal references a non-serialized function by f"{module}:{qualname}"
        # and re-imports that module in the container. @wraps put the activity's
        # module on `runner`; bind it there as a real global so the lookup
        # resolves. The decorator re-runs on the remote import and re-binds it.
        setattr(sys.modules[raw_f.__module__], modal_name, runner)

        if partial is not None and inner is not None:
            # Redirect the PartialFunction's raw_f to our runner so that Modal
            # settings like @modal.concurrent carry through to the runner.
            inner.raw_f = runner
            app.function(**modal_opts)(partial)
        else:
            app.function(**modal_opts)(runner)

        REGISTRY[raw_f.__name__] = _Runner(modal_name, is_class=False)
        return temporal_activity

    return decorate


_ACTIVITY_METHOD = "__modal_activity_method__"


def modal_activity_method(fn: Callable[P, R]) -> Callable[P, R]:
    """Mark a method of a @modal_activity_cls class as a Temporal activity.
    Applies a bare @activity.defn (name == method __name__) and tags it so the
    class decorator selects exactly these methods."""
    temporal_method: Callable[P, R] = activity.defn(fn)
    setattr(temporal_method, _ACTIVITY_METHOD, True)
    return temporal_method


# modal.parameter() returns a modal.cls._Parameter whose .default is a
# _NO_DEFAULT sentinel instance when no default was given.
_MODAL_PARAM = type(modal.parameter())
_MODAL_NO_DEFAULT = type(modal.parameter().default)


@dataclass_transform(field_specifiers=(modal.parameter,))
def modal_activity_cls(
    app: modal.App, **modal_opts: Any
) -> Callable[[Union[type[T], _ModalPartialFunction[Any, T, T]]], type[T]]:
    """Class analogue of @modal_activity. Decorate the activity class itself: a
    runner @app.cls() is generated from its __init__ params and public async
    methods, bound into the activity's module so Modal can reference it by
    symbol (no serialized=True; the decorator re-runs on the remote import and
    re-binds it), and recorded in REGISTRY. Returns the class unchanged.

    Params may be declared either with a normal __init__ or Modal-style as
    `name: T = modal.parameter()`; the latter is turned into a dataclass
    __init__ so the activity stays a normal, instantiable class.

    Assumes bare @activity.defn on the methods => activity name == method
    __name__ (same convention as @modal_activity)."""

    def decorate(cls: Union[type[T], _ModalPartialFunction[Any, T, T]]) -> type[T]:
        # Unwrap Modal PartialFunction (e.g. from @modal.concurrent) to get
        # the plain class that the rest of this decorator requires.
        # When wrapping a class, PartialFunction stores it in user_cls (not raw_f).
        partial = None
        inner = None
        if isinstance(cls, _ModalPartialFunction):
            inner = getattr(cls, getattr(cls, "_sync_synchronizer")._original_attr)
            partial = cls
            cls = cast(type[T], inner.user_cls)

        # Modal-style params: synthesize a dataclass __init__ so the queuer can
        # build an instance and the interceptor can read vars(instance). The
        # signature logic below is then identical for both styles.
        if "__init__" not in cls.__dict__:
            cls_anns = dict(getattr(cls, "__annotations__", {}))
            cls_attrs = dict(vars(cls))
            params = {
                n: cls_attrs[n]
                for n in cls_anns
                if isinstance(cls_attrs.get(n), _MODAL_PARAM)
            }
            if params:
                for n, pobj in params.items():
                    if isinstance(pobj.default, _MODAL_NO_DEFAULT):
                        delattr(cls, n)  # required dataclass field
                    else:
                        setattr(cls, n, pobj.default)
                dataclasses.dataclass(cls)

        # Dynamically add `modal.parameters`
        anns: dict[str, Any] = {}
        ns: dict[str, Any] = {}
        param_names: list[str] = []
        for p in inspect.signature(cls.__init__).parameters.values():
            if p.name == "self" or p.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue
            param_names.append(p.name)
            anns[p.name] = (
                p.annotation if p.annotation is not inspect.Parameter.empty else Any
            )
            ns[p.name] = (
                modal.parameter()
                if p.default is inspect.Parameter.empty
                else modal.parameter(default=p.default)
            )
        ns["__annotations__"] = anns

        # Dynamically add a `modal.enter()`
        async def _start(self):
            self._activity = cls(**{n: getattr(self, n) for n in param_names})

        ns["start"] = modal.enter()(_start)

        # Dynamically add methods from activities
        def make_method(method_name: str):
            async def _run(self, info: Info, args: Any):
                client = await get_temporal_client()
                return await run_activity(
                    getattr(self._activity, method_name), args, client, info
                )

            return modal.method()(_run)

        methods = [
            n
            for n, fn in dict(vars(cls)).items()
            if getattr(fn, _ACTIVITY_METHOD, False)
        ]
        for n in methods:
            ns[n] = make_method(n)

        runner_name = f"{cls.__name__}Runner"
        runner_cls = type(runner_name, (), ns)
        # Same trick as @modal_activity: make the generated runner a real
        # f"{module}:{qualname}" symbol in the activity's module so Modal's
        # by-reference lookup resolves. type() set __module__ to this module;
        # point it at the activity's module instead.
        runner_cls.__module__ = cls.__module__
        runner_cls.__qualname__ = runner_name
        setattr(sys.modules[cls.__module__], runner_name, runner_cls)

        if inner is not None and partial is not None:
            # Redirect the PartialFunction to wrap runner_cls so Modal settings
            # like @modal.concurrent carry through to the runner class.
            inner.user_cls = runner_cls
            app.cls(**modal_opts)(partial)
        else:
            app.cls(**modal_opts)(runner_cls)

        for n in methods:
            REGISTRY[n] = _Runner(runner_name, is_class=True)
        return cls

    return decorate


@alru_cache()
async def get_modal_function(app_name: str, func_name: str) -> modal.Function:
    return await modal.Function.from_name(app_name, func_name).hydrate.aio()


@alru_cache()
async def get_modal_cls(app_name: str, cls_name: str) -> modal.Cls:
    return await modal.Cls.from_name(app_name, cls_name).hydrate.aio()


class DispatchActivityInterceptor(ActivityInboundInterceptor):
    def __init__(self, next: ActivityInboundInterceptor, app_name: str) -> None:
        super().__init__(next)
        self._app_name = app_name

    async def execute_activity(self, input: ExecuteActivityInput) -> Any:
        info = activity.info()
        args = list(input.args)

        entry = REGISTRY[info.activity_type]

        if entry.is_class:
            # input.fn is a bound method, e.g. SayHello().run. The instance's
            # attributes map to the Modal Cls parameters.
            assert isinstance(input.fn, types.MethodType)
            instance = input.fn.__self__
            modal_cls = await get_modal_cls(self._app_name, entry.modal_name)
            modal_handle = getattr(modal_cls(**vars(instance)), input.fn.__name__)
        else:
            modal_handle = await get_modal_function(self._app_name, entry.modal_name)

        # Heartbeat while the activity sits on Modal's input queue. Seed the
        # coordination key BEFORE spawning so a fast-starting worker's
        # started:<attempt> marker is never clobbered by this queued marker.
        if info.heartbeat_timeout:
            await self._start_queue_heartbeat(info)

        print(f"[dispatcher] {info.activity_type} -> {entry.modal_name} args={args}")
        await modal_handle.spawn.aio(info, args)
        activity.raise_complete_async()

    async def _start_queue_heartbeat(self, info: Info) -> None:
        client = await get_temporal_client()
        coord_dict = await get_heartbeat_coord_dict()
        await coord_dict.put.aio(_coord_key(info), _queued_marker(info))
        handle = client.get_async_activity_handle(task_token=info.task_token)
        task = asyncio.create_task(
            _dispatcher_heartbeat_loop(handle, info, coord_dict)
        )
        _dispatcher_heartbeat_tasks.add(task)
        task.add_done_callback(_dispatcher_heartbeat_tasks.discard)


class DispatchInterceptor(Interceptor):
    def __init__(self, app_name: str) -> None:
        self._app_name = app_name

    def intercept_activity(
        self, next: ActivityInboundInterceptor
    ) -> ActivityInboundInterceptor:
        return DispatchActivityInterceptor(next, self._app_name)


async def run_dispatcher(
    app_name: str,
    *,
    task_queue: str,
    workflows: Sequence[type],
    activities: Sequence[Callable],
    max_workers: int = 4,
    worker_kwargs: dict | None = None,
) -> None:
    """Run the Temporal worker that dispatches activities to Modal.

    Owns only the dispatcher plumbing: the Temporal client, the
    DispatchInterceptor, and a thread pool so sync activities can run. Extra
    keyword args are forwarded to temporalio's Worker, so callers can set any
    Worker parameter. Special-cased so the plumbing is not lost:
    - `interceptors` you pass are appended after the DispatchInterceptor.
    - `activity_executor`, if given, replaces the default thread pool.
    """
    if worker_kwargs is None:
        worker_kwargs = {}
    client = await get_temporal_client()
    interceptors = [
        DispatchInterceptor(app_name),
        *worker_kwargs.pop("interceptors", []),
    ]
    worker_kwargs.setdefault(
        "activity_executor", ThreadPoolExecutor(max_workers=max_workers)
    )
    worker = Worker(
        client,
        task_queue=task_queue,
        workflows=list(workflows),
        activities=list(activities),
        interceptors=interceptors,
        **worker_kwargs,
    )
    print(f"[dispatcher] worker on {task_queue!r}; activities run on Modal")
    await worker.run()
