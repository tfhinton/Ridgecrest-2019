'''Shared loader for profile_inversion/*/profile_*.pickle result files.

arviz >=1.0 was a from-scratch rewrite (InferenceData is now backed by an
xarray DataTree) that dropped the old `arviz.data.inference_data` module, so
`pickle.load` on a result written with an older arviz raises
`ModuleNotFoundError: No module named 'arviz.data'` -- even though the actual
posterior/sample_stats groups it stored (plain xarray.Dataset objects) are
unaffected and still deserialize fine. Rather than re-run finished (possibly
multi-hour cluster) inversions or pin an old arviz, load_result() substitutes
a bare shim class for the one broken outer wrapper, which is enough to
recover `idata.posterior` / `idata.sample_stats` exactly as before.
'''
import pickle


class _Shim:
    def __new__(cls, *a, **k):
        return object.__new__(cls)

    def __setstate__(self, state):
        self.__dict__.update(state)


class _ShimUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except ModuleNotFoundError:
            shim = type(name, (_Shim,), {})
            shim.__module__ = module
            return shim


def load_result(path):
    '''Load a result pickle, tolerating the arviz InferenceData break above.'''
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except ModuleNotFoundError:
        with open(path, 'rb') as f:
            return _ShimUnpickler(f).load()


def param_labels(r):
    '''Canonical free-parameter order for a loaded result. Older result
    pickles (pre-dating a `param_labels` field) don't have the key, but
    `best` (built as dict(zip(param_labels, ...))) preserves the same order.'''
    return r.get('param_labels') or list(r['best'].keys())
