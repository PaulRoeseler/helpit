from __future__ import annotations

import dataclasses
import inspect
import reprlib
import sys
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Dict, List, Optional

MODULE_TO_DIST: Dict[str, List[str]] = {
    "PIL": ["Pillow"],
    "sklearn": ["scikit-learn"],
    "cv2": ["opencv-python", "opencv-contrib-python"],
    "yaml": ["PyYAML"],
    "dateutil": ["python-dateutil"],
    "Crypto": ["pycryptodome", "pycrypto"],
    "bs4": ["beautifulsoup4"],
    "setuptools": ["setuptools"],
}


def _full_type_name(obj: Any) -> str:
    t = type(obj)
    return f"{t.__module__}.{t.__qualname__}"


def _type_module(obj: Any) -> str:
    return type(obj).__module__ or ""


def _top_module(type_module: str) -> str:
    return type_module.split(".", 1)[0] if type_module else ""


def _short_repr(obj: Any, max_chars: int) -> str:
    r = reprlib.Repr()
    r.maxstring = max_chars
    r.maxother = max_chars
    r.maxlist = 10
    r.maxtuple = 10
    r.maxset = 10
    r.maxdict = 10
    r.maxbytes = max_chars
    try:
        return r.repr(obj)
    except Exception:
        return f"<unreprable {_full_type_name(obj)}>"


def _safe_len(obj: Any) -> Optional[int]:
    try:
        return len(obj)
    except Exception:
        return None


def _is_iterable(obj: Any) -> bool:
    try:
        iter(obj)
        return True
    except Exception:
        return False


def _safe_signature(callable_obj: Any) -> Optional[str]:
    try:
        return str(inspect.signature(callable_obj))
    except Exception:
        return None


def _public_attr_names_sample(obj: Any, max_attrs: int) -> Optional[List[str]]:
    try:
        names = [n for n in dir(obj) if not n.startswith("_")]
        return names[:max_attrs]
    except Exception:
        return None


def _dist_candidates_for_top_module(top_module: str) -> List[str]:
    if not top_module or top_module in ("builtins", "__main__"):
        return []
    return MODULE_TO_DIST.get(top_module, []) + [top_module]


def _distribution_info(top_module: str) -> Dict[str, Optional[str]]:
    for dist in _dist_candidates_for_top_module(top_module):
        try:
            ver = importlib_metadata.version(dist)
            return {"distribution_name_guess": dist, "distribution_version": ver}
        except Exception:
            continue
    return {"distribution_name_guess": None, "distribution_version": None}


def _popular_library_extras(obj: Any, *, max_names: int) -> Dict[str, Any]:
    extras: Dict[str, Any] = {}
    tm = _type_module(obj)
    top = _top_module(tm)

    for attr in ("shape", "dtype", "ndim", "size"):
        try:
            v = getattr(obj, attr)
        except Exception:
            continue
        else:
            try:
                if attr == "shape":
                    extras["shape"] = tuple(v)
                elif attr == "ndim":
                    extras["ndim"] = int(v)
                elif attr == "size":
                    extras["size"] = int(v)
                else:
                    extras["dtype"] = str(v)
            except Exception:
                continue

    if top == "torch":
        for attr in ("device", "requires_grad"):
            if hasattr(obj, attr):
                try:
                    extras[attr] = str(getattr(obj, attr)) if attr == "device" else bool(getattr(obj, attr))
                except Exception:
                    pass

    if top == "pandas":
        if hasattr(obj, "columns"):
            try:
                cols = list(getattr(obj, "columns"))
                extras["columns_count"] = len(cols)
                extras["columns_sample"] = [str(c) for c in cols[:max_names]]
            except Exception:
                pass
        if hasattr(obj, "dtypes"):
            try:
                extras["dtypes_repr"] = _short_repr(getattr(obj, "dtypes"), 200)
            except Exception:
                pass
        if hasattr(obj, "index"):
            try:
                idx = getattr(obj, "index")
                extras["index_type"] = f"{type(idx).__module__}.{type(idx).__qualname__}"
                extras["index_len"] = _safe_len(idx)
            except Exception:
                pass

    if top in ("PIL", "Pillow"):
        for attr in ("size", "mode", "format"):
            if hasattr(obj, attr):
                try:
                    extras[attr] = str(getattr(obj, attr))
                except Exception:
                    pass

    if isinstance(obj, Path):
        extras["path_name"] = obj.name
        extras["path_suffix"] = obj.suffix
        extras["path_parent"] = str(obj.parent)

    if top == "sklearn" and hasattr(obj, "get_params"):
        try:
            params = obj.get_params(deep=False)
            if isinstance(params, dict):
                extras["params_keys_sample"] = list(params.keys())[:max_names]
        except Exception:
            pass

    return extras


def object_header(
    obj: Any,
    *,
    max_repr_chars: int = 1000,
    max_attrs: int = 30,
    max_names: int = 30,
    include_public_attrs_sample: bool = True,
) -> Dict[str, Any]:
    tm = _type_module(obj)
    top = _top_module(tm)
    dist = _distribution_info(top)

    header: Dict[str, Any] = {
        "schema": "objhdr.v2",
        "python": sys.version.split()[0],
        "type": _full_type_name(obj),
        "type_module": tm,
        "top_module": top,
        **dist,
        "is_callable": callable(obj),
        "is_iterable": _is_iterable(obj),
    }

    length = _safe_len(obj)
    if length is not None:
        header["len"] = length

    if isinstance(obj, (str, bytes, bytearray)):
        header["value_kind"] = type(obj).__name__
        header["value_len"] = len(obj)
        header["repr"] = f"<{type(obj).__name__} redacted>"
        return header

    if header["is_callable"]:
        header["callable_name"] = getattr(obj, "__qualname__", getattr(obj, "__name__", None))
        sig = _safe_signature(obj)
        if sig:
            header["signature"] = sig

        bound_self = getattr(obj, "__self__", None)
        if bound_self is not None:
            header["bound_self_type"] = _full_type_name(bound_self)
            header["bound_self_type_module"] = _type_module(bound_self)
            header["bound_self_top_module"] = _top_module(_type_module(bound_self))
            try:
                header["bound_self_extras"] = _popular_library_extras(bound_self, max_names=max_names)
            except Exception:
                pass

    try:
        if dataclasses.is_dataclass(obj):
            header["dataclass_fields"] = [f.name for f in dataclasses.fields(obj)][:max_attrs]
    except Exception:
        pass

    try:
        mf = getattr(obj, "model_fields", None)
        if isinstance(mf, dict):
            header["pydantic_fields"] = list(mf.keys())[:max_attrs]
    except Exception:
        pass

    try:
        f = getattr(obj, "__fields__", None)
        if isinstance(f, dict):
            header["pydantic_fields"] = list(f.keys())[:max_attrs]
    except Exception:
        pass

    if include_public_attrs_sample:
        attrs = _public_attr_names_sample(obj, max_attrs=max_attrs)
        if attrs:
            header["public_attrs_sample"] = attrs

    try:
        header.update(_popular_library_extras(obj, max_names=max_names))
    except Exception:
        pass

    header["repr"] = _short_repr(obj, max_repr_chars)
    return header


__all__ = [
    "MODULE_TO_DIST",
    "object_header",
]
