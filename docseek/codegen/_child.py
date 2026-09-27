"""The sandbox child: runs one generated discovery program. Started by docseek.codegen.sandbox as a plain script
(`python _child.py`), never imported: it uses only the standard library, so it neither loads docseek (and its
.env) nor re-runs the caller's main module, as multiprocessing would.

Protocol, one JSON object per line. The parent writes {"code", "cpu_seconds", "allowed", "max_documents"} first,
then answers each request with {"ok": bool, "value": str}. The child writes {"op": "fetch"|"render", "url"} or
{"op": "post", "url", "body"} for pages, and ends with {"op": "done", "documents": [...]} or
{"op": "error", "message"}. The program's own prints go to stderr, so they cannot break the protocol.
"""
import builtins
import importlib
import json
import os
import sys
import traceback

_BLOCKED_EVENTS = ('socket.', 'subprocess.', 'os.system', 'os.exec', 'os.spawn', 'os.posix_spawn', 'os.fork',
                   'open', 'ctypes.', 'os.remove', 'os.rename', 'os.putenv', 'shutil.')
_BLOCKED_BUILTINS = ('open', 'exec', 'eval', 'compile', 'input', 'breakpoint', 'exit', 'quit', 'help')
MEMORY_LIMIT_BYTES = 2 * 1024 ** 3


class FetchRefused(Exception):
    """The parent's fetcher said no; the message says why."""


def _limit_resources(cpu_seconds: int) -> None:
    try:
        import resource
    except ImportError:                                   # not a POSIX system: the parent's timeout still applies
        return
    for limit, value in ((resource.RLIMIT_CPU, cpu_seconds), (getattr(resource, 'RLIMIT_AS', None), MEMORY_LIMIT_BYTES)):
        if limit is None:
            continue
        try:
            resource.setrlimit(limit, (value, value))
        except (ValueError, OSError):                     # macOS refuses RLIMIT_AS; keep what the platform allows
            pass


def describe_error(exc: BaseException, code: str) -> str:
    """The traceback without reading source files (the audit hook refuses open): the program's own lines are
    quoted from its code, library frames by name."""
    lines = code.splitlines()
    frames = []
    for frame, lineno in traceback.walk_tb(exc.__traceback__):
        name = frame.f_code.co_filename
        where = f'{name}:{lineno} in {frame.f_code.co_name}'
        if name == 'discovery_program.py' and 0 < lineno <= len(lines):
            where += f'\n    {lines[lineno - 1].strip()}'
        frames.append(where)
    return '\n'.join(frames[-6:] + [f'{type(exc).__name__}: {exc}'])[-3000:]


def main() -> None:
    protocol_out, protocol_in = sys.stdout, sys.stdin
    sys.stdout = sys.stderr                               # the program's prints must not reach the protocol

    def send(message: dict) -> None:
        protocol_out.write(json.dumps(message) + '\n')
        protocol_out.flush()

    setup = json.loads(protocol_in.readline())
    code, allowed = setup['code'], tuple(setup['allowed'])

    modules = {name: importlib.import_module(name) for name in allowed}
    modules['urllib'] = importlib.import_module('urllib')
    real_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name in modules or any(name == m.split('.')[0] for m in allowed):
            return real_import(name, globals, locals, fromlist, level)
        raise ImportError(f'module {name!r} is not available to a discovery program')

    def audit(event, args):
        if event.startswith(_BLOCKED_EVENTS):
            raise RuntimeError(f'{event} is not allowed in a discovery program')

    def ask(request: dict) -> str:
        send(request)
        answer = json.loads(protocol_in.readline())
        if not answer['ok']:
            raise FetchRefused(answer['value'])
        return answer['value']

    safe_builtins = {k: getattr(builtins, k) for k in dir(builtins) if k not in _BLOCKED_BUILTINS}
    safe_builtins['__import__'] = guarded_import
    env = {'__builtins__': safe_builtins, '__name__': 'discovery_program', 'FetchRefused': FetchRefused}
    try:
        compiled = compile(code, 'discovery_program.py', 'exec')
        os.environ.clear()                                # nothing the parent's environment held stays reachable
        _limit_resources(int(setup['cpu_seconds']))
        sys.addaudithook(audit)
        exec(compiled, env)
        discover = env['discover']
        args = [lambda url: ask({'op': 'fetch', 'url': url}), lambda url: ask({'op': 'render', 'url': url})]
        if discover.__code__.co_argcount >= 3:
            args.append(lambda url, body: ask({'op': 'post', 'url': url, 'body': body}))
        found = discover(*args)
        docs = [{'url': str(d['url']), 'name': str(d.get('name', ''))[:300], 'context': str(d.get('context', ''))[:400]}
                for d in list(found)[:int(setup['max_documents'])] if isinstance(d, dict) and d.get('url')]
        send({'op': 'done', 'documents': docs})
    except BaseException as exc:  # noqa: BLE001 - the program's own failure is the result
        send({'op': 'error', 'message': describe_error(exc, code)})


if __name__ == '__main__':
    main()
