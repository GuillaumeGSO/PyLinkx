import base64
import json

import numpy as np

_ORT_CDN = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/"


def _js_query(expr):
    """Evaluate a single JS expression, return Python value via JSON roundtrip."""
    import embed
    result = embed.run_script(f"JSON.stringify({expr})")
    if result is not None:
        return json.loads(result)
    return None


def _js_exec(code):
    """Execute JS code block, fire-and-forget."""
    import embed
    embed.run_script(code)


class WasmModelLoader:
    """Step-by-step ONNX model loader for pygbag/WASM.

    Call step() once per game-loop frame.  Each call performs one
    synchronous JS operation, then returns so the game loop can
    yield to the browser via its own ``await asyncio.sleep()``.
    """

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.status = "read"
        self.label = "Reading model..."
        self.error = None
        self._wait = 0

    @property
    def done(self) -> bool:
        return self.status == "done"

    def step(self):
        try:
            self._step_inner()
        except Exception as e:
            self.error = str(e)

    def _step_inner(self):
        if self.status == "read":
            with open(self.model_path, "rb") as f:
                self._model_bytes = f.read()
            self.label = "Loading AI engine..."
            self.status = "inject"

        elif self.status == "inject":
            if _js_query("typeof ort !== 'undefined'"):
                self.status = "configure"
            else:
                _js_exec(
                    "var s = document.createElement('script');"
                    f"s.src = '{_ORT_CDN}ort.wasm.min.js';"
                    "document.head.appendChild(s)"
                )
                self._wait = 0
                self.status = "wait_ort"

        elif self.status == "wait_ort":
            self._wait += 1
            if _js_query("typeof ort !== 'undefined'"):
                self.status = "configure"
            elif self._wait > 100:
                self.error = "Timeout loading AI engine from CDN"

        elif self.status == "configure":
            _js_exec(f"ort.env.wasm.wasmPaths = '{_ORT_CDN}'")
            _js_exec("ort.env.wasm.numThreads = 1")
            self.label = "Creating AI session..."
            self.status = "create"

        elif self.status == "create":
            b64 = base64.b64encode(self._model_bytes).decode()
            self._model_bytes = None  # free memory
            _js_exec(
                "window._pylinkx_session = null;"
                "window._pylinkx_session_error = null;"
                "(function(){"
                f"var b = atob('{b64}');"
                "var a = new Uint8Array(b.length);"
                "for(var i=0;i<b.length;i++) a[i]=b.charCodeAt(i);"
                "ort.InferenceSession.create(a)"
                ".then(function(s){window._pylinkx_session=s})"
                ".catch(function(e){"
                "window._pylinkx_session_error=e.message||String(e)})"
                "})()"
            )
            self._wait = 0
            self.status = "wait_session"

        elif self.status == "wait_session":
            self._wait += 1
            if _js_query("window._pylinkx_session !== null"):
                self.status = "done"
            else:
                err = _js_query("window._pylinkx_session_error")
                if err:
                    self.error = f"AI session failed: {err}"
                elif self._wait > 300:
                    self.error = "Timeout creating AI session"


class WasmOnnxPolicy:
    """ONNX inference via onnxruntime-web (pygbag/WASM only).

    Session lives on ``window._pylinkx_session``; predict dispatches
    inference to JS and polls for the result.
    """

    def __init__(self):
        pass

    async def predict(self, obs, action_masks=None, deterministic=True):
        import asyncio

        grid = obs["grid"][np.newaxis].astype(np.float32)
        scalars = obs["scalars"][np.newaxis].astype(np.float32)

        grid_b64 = base64.b64encode(grid.tobytes()).decode()
        scalars_b64 = base64.b64encode(scalars.tobytes()).decode()

        _js_exec(
            "window._pylinkx_result = null;"
            "window._pylinkx_result_error = null;"
            "(function(){"
            "function d(b64,shape){"
            "var b=atob(b64);"
            "var buf=new ArrayBuffer(b.length);"
            "var v=new Uint8Array(buf);"
            "for(var i=0;i<b.length;i++) v[i]=b.charCodeAt(i);"
            "return new ort.Tensor('float32',new Float32Array(buf),shape)}"
            "window._pylinkx_session.run({"
            f"grid:d('{grid_b64}',[1,9,9,1]),"
            f"scalars:d('{scalars_b64}',[1,258])"
            "}).then(function(r){"
            "window._pylinkx_result=Array.from(r.logits.data)"
            "}).catch(function(e){"
            "window._pylinkx_result_error=e.message||String(e)"
            "})"
            "})()"
        )

        for _ in range(100):
            await asyncio.sleep(0.05)
            result = _js_query("window._pylinkx_result")
            if result is not None and result is not False:
                break
            err = _js_query("window._pylinkx_result_error")
            if err:
                raise RuntimeError(f"ONNX inference failed: {err}")
        else:
            raise RuntimeError("Timeout during ONNX inference")

        logits = np.array(result, dtype=np.float32)

        if action_masks is not None:
            logits[~action_masks.astype(bool)] = -np.inf
        if deterministic:
            return int(np.argmax(logits)), None
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        return int(np.random.choice(len(probs), p=probs)), None
