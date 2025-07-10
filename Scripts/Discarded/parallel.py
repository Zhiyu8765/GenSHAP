{\rtf1\ansi\ansicpg1252\cocoartf2822
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fmodern\fcharset0 Courier-Oblique;\f1\fmodern\fcharset0 Courier;}
{\colortbl;\red255\green255\blue255;\red78\green112\blue88;\red23\green23\blue26;\red103\green107\blue114;
\red195\green123\blue90;\red174\green176\blue183;\red117\green114\blue185;\red89\green158\blue96;\red71\green149\blue242;
\red152\green54\blue29;\red38\green157\blue169;\red160\green0\blue163;\red128\green63\blue122;}
{\*\expandedcolortbl;;\csgenericrgb\c30588\c43922\c34510;\csgenericrgb\c9020\c9020\c10196;\csgenericrgb\c40392\c41961\c44706;
\csgenericrgb\c76471\c48235\c35294;\csgenericrgb\c68235\c69020\c71765;\csgenericrgb\c45882\c44706\c72549;\csgenericrgb\c34902\c61961\c37647;\csgenericrgb\c27843\c58431\c94902;
\csgenericrgb\c59608\c21176\c11373;\csgenericrgb\c14902\c61569\c66275;\csgenericrgb\c62745\c0\c63922;\csgenericrgb\c50196\c24706\c47843;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx560\tx1120\tx1680\tx2240\tx2800\tx3360\tx3920\tx4480\tx5040\tx5600\tx6160\tx6720\pardirnatural\partightenfactor0

\f0\i\fs30 \cf2 \cb3 """\
Helpers for embarrassingly parallel code.\
"""\

\f1\i0 \cf4 # Author: Gael Varoquaux < gael dot varoquaux at normalesup dot org >\
# Copyright: 2010, Gael Varoquaux\
# License: BSD 3 clause\
\
\cf5 from \cf6 __future__ \cf5 import \cf6 division\
\
\cf5 import \cf6 os\
\cf5 import \cf6 sys\
\cf5 from \cf6 math \cf5 import \cf6 sqrt\
\cf5 import \cf6 functools\
\cf5 import \cf6 collections\
\cf5 import \cf6 time\
\cf5 import \cf6 threading\
\cf5 import \cf6 itertools\
\cf5 from \cf6 uuid \cf5 import \cf6 uuid4\
\cf5 from \cf6 numbers \cf5 import \cf6 Integral\
\cf5 import \cf6 warnings\
\cf5 import \cf6 queue\
\cf5 import \cf6 weakref\
\cf5 from \cf6 contextlib \cf5 import \cf6 nullcontext\
\
\cf5 from \cf6 multiprocessing \cf5 import \cf6 TimeoutError\
\
\cf5 from \cf6 ._multiprocessing_helpers \cf5 import \cf6 mp\
\
\cf5 from \cf6 .logger \cf5 import \cf6 Logger, short_format_time\
\cf5 from \cf6 .disk \cf5 import \cf6 memstr_to_bytes\
\cf5 from \cf6 ._parallel_backends \cf5 import \cf6 (FallbackToBackend, MultiprocessingBackend,\
                                 ThreadingBackend, SequentialBackend,\
                                 LokyBackend)\
\cf5 from \cf6 ._utils \cf5 import \cf6 eval_expr, _Sentinel\
\
\cf4 # Make sure that those two classes are part of the public joblib.parallel API\
# so that 3rd party backend implementers can import them from here.\
\cf5 from \cf6 ._parallel_backends \cf5 import \cf6 AutoBatchingMixin  \cf4 # noqa\
\cf5 from \cf6 ._parallel_backends \cf5 import \cf6 ParallelBackendBase  \cf4 # noqa\
\
\
\cf6 IS_PYPY = \cf7 hasattr\cf6 (sys, \cf8 "pypy_version_info"\cf6 )\
\
\
BACKENDS = \{\
    \cf8 'threading'\cf6 : ThreadingBackend,\
    \cf8 'sequential'\cf6 : SequentialBackend,\
\}\
\cf4 # name of the backend used by default by Parallel outside of any context\
# managed by ``parallel_config`` or ``parallel_backend``.\
\
# threading is the only backend that is always everywhere\
\cf6 DEFAULT_BACKEND = \cf8 'threading'\
\
\cf6 MAYBE_AVAILABLE_BACKENDS = \{\cf8 'multiprocessing'\cf6 , \cf8 'loky'\cf6 \}\
\
\cf4 # if multiprocessing is available, so is loky, we set it as the default\
# backend\
\cf5 if \cf6 mp \cf5 is not None\cf6 :\
    BACKENDS[\cf8 'multiprocessing'\cf6 ] = MultiprocessingBackend\
    \cf5 from \cf6 .externals \cf5 import \cf6 loky\
    BACKENDS[\cf8 'loky'\cf6 ] = LokyBackend\
    DEFAULT_BACKEND = \cf8 'loky'\
\
\
\cf6 DEFAULT_THREAD_BACKEND = \cf8 'threading'\
\
\
\cf4 # Thread local value that can be overridden by the ``parallel_config`` context\
# manager\
\cf6 _backend = threading.local()\
\
\
\cf5 def \cf9 _register_dask\cf6 ():\
    
\f0\i \cf2 """Register Dask Backend if called with parallel_config(backend="dask")"""\
    
\f1\i0 \cf5 try\cf6 :\
        \cf5 from \cf6 ._dask \cf5 import \cf6 DaskDistributedBackend\
        register_parallel_backend(\cf8 'dask'\cf6 , DaskDistributedBackend)\
    \cf5 except \cf7 ImportError \cf5 as \cf6 e:\
        msg = (\cf8 "To use the dask.distributed backend you must install both "\
               "the `dask` and distributed modules.\cf5 \\n\\n\cf8 "\
               "See https://dask.pydata.org/en/latest/install.html for more "\
               "information."\cf6 )\
        \cf5 raise \cf7 ImportError\cf6 (msg) \cf5 from \cf6 e\
\
\
EXTERNAL_BACKENDS = \{\
    \cf8 'dask'\cf6 : _register_dask,\
\}\
\
\
\cf4 # Sentinels for the default values of the Parallel constructor and\
# the parallel_config and parallel_backend context managers\
\cf6 default_parallel_config = \{\
    \cf8 "backend"\cf6 : _Sentinel(\cf10 default_value\cf6 =\cf5 None\cf6 ),\
    \cf8 "n_jobs"\cf6 : _Sentinel(\cf10 default_value\cf6 =\cf5 None\cf6 ),\
    \cf8 "verbose"\cf6 : _Sentinel(\cf10 default_value\cf6 =\cf11 0\cf6 ),\
    \cf8 "temp_folder"\cf6 : _Sentinel(\cf10 default_value\cf6 =\cf5 None\cf6 ),\
    \cf8 "max_nbytes"\cf6 : _Sentinel(\cf10 default_value\cf6 =\cf8 "1M"\cf6 ),\
    \cf8 "mmap_mode"\cf6 : _Sentinel(\cf10 default_value\cf6 =\cf8 "r"\cf6 ),\
    \cf8 "prefer"\cf6 : _Sentinel(\cf10 default_value\cf6 =\cf5 None\cf6 ),\
    \cf8 "require"\cf6 : _Sentinel(\cf10 default_value\cf6 =\cf5 None\cf6 ),\
\}\
\
\
VALID_BACKEND_HINTS = (\cf8 'processes'\cf6 , \cf8 'threads'\cf6 , \cf5 None\cf6 )\
VALID_BACKEND_CONSTRAINTS = (\cf8 'sharedmem'\cf6 , \cf5 None\cf6 )\
\
\
\cf5 def \cf9 _get_config_param\cf6 (param, context_config, key):\
    
\f0\i \cf2 """Return the value of a parallel config parameter\
\
    Explicitly setting it in Parallel has priority over setting in a\
    parallel_(config/backend) context manager.\
    """\
    
\f1\i0 \cf5 if \cf6 param \cf5 is not \cf6 default_parallel_config[key]:\
        \cf4 # param is explicitely set, return it\
        \cf5 return \cf6 param\
\
    \cf5 if \cf6 context_config[key] \cf5 is not \cf6 default_parallel_config[key]:\
        \cf4 # there's a context manager and the key is set, return it\
        \cf5 return \cf6 context_config[key]\
\
    \cf4 # Otherwise, we are in the default_parallel_config,\
    # return the default value\
    \cf5 return \cf6 param.default_value\
\
\
\cf5 def \cf9 get_active_backend\cf6 (\
    prefer=default_parallel_config[\cf8 "prefer"\cf6 ],\
    require=default_parallel_config[\cf8 "require"\cf6 ],\
    verbose=default_parallel_config[\cf8 "verbose"\cf6 ],\
):\
    
\f0\i \cf2 """Return the active default backend"""\
    
\f1\i0 \cf6 backend, config = _get_active_backend(prefer, require, verbose)\
    n_jobs = _get_config_param(\
        default_parallel_config[\cf8 'n_jobs'\cf6 ], config, \cf8 "n_jobs"\
    \cf6 )\
    \cf5 return \cf6 backend, n_jobs\
\
\
\cf5 def \cf9 _get_active_backend\cf6 (\
    prefer=default_parallel_config[\cf8 "prefer"\cf6 ],\
    require=default_parallel_config[\cf8 "require"\cf6 ],\
    verbose=default_parallel_config[\cf8 "verbose"\cf6 ],\
):\
    
\f0\i \cf2 """Return the active default backend"""\
\
    
\f1\i0 \cf6 backend_config = \cf7 getattr\cf6 (_backend, \cf8 "config"\cf6 , default_parallel_config)\
\
    backend = _get_config_param(\
        default_parallel_config[\cf8 'backend'\cf6 ], backend_config, \cf8 "backend"\
    \cf6 )\
    prefer = _get_config_param(prefer, backend_config, \cf8 "prefer"\cf6 )\
    require = _get_config_param(require, backend_config, \cf8 "require"\cf6 )\
    verbose = _get_config_param(verbose, backend_config, \cf8 "verbose"\cf6 )\
\
    \cf5 if \cf6 prefer \cf5 not in \cf6 VALID_BACKEND_HINTS:\
        \cf5 raise \cf7 ValueError\cf6 (\
            \cf8 f"prefer=\cf5 \{\cf6 prefer\cf5 \}\cf8  is not a valid backend hint, "\
            f"expected one of \cf5 \{\cf6 VALID_BACKEND_HINTS\cf5 \}\cf8 "\
        \cf6 )\
    \cf5 if \cf6 require \cf5 not in \cf6 VALID_BACKEND_CONSTRAINTS:\
        \cf5 raise \cf7 ValueError\cf6 (\
            \cf8 f"require=\cf5 \{\cf6 require\cf5 \}\cf8  is not a valid backend constraint, "\
            f"expected one of \cf5 \{\cf6 VALID_BACKEND_CONSTRAINTS\cf5 \}\cf8 "\
        \cf6 )\
    \cf5 if \cf6 prefer == \cf8 'processes' \cf5 and \cf6 require == \cf8 'sharedmem'\cf6 :\
        \cf5 raise \cf7 ValueError\cf6 (\
            \cf8 "prefer == 'processes' and require == 'sharedmem'"\
            " are inconsistent settings"\
        \cf6 )\
\
    explicit_backend = \cf5 True\
    if \cf6 backend \cf5 is None\cf6 :\
\
        \cf4 # We are either outside of the scope of any parallel_(config/backend)\
        # context manager or the context manager did not set a backend.\
        # create the default backend instance now.\
        \cf6 backend = BACKENDS[DEFAULT_BACKEND](\cf10 nesting_level\cf6 =\cf11 0\cf6 )\
        explicit_backend = \cf5 False\
\
    \cf4 # Try to use the backend set by the user with the context manager.\
\
    \cf6 nesting_level = backend.nesting_level\
    uses_threads = \cf7 getattr\cf6 (backend, \cf8 'uses_threads'\cf6 , \cf5 False\cf6 )\
    supports_sharedmem = \cf7 getattr\cf6 (backend, \cf8 'supports_sharedmem'\cf6 , \cf5 False\cf6 )\
    \cf4 # Force to use thread-based backend if the provided backend does not\
    # match the shared memory constraint or if the backend is not explicitely\
    # given and threads are prefered.\
    \cf6 force_threads = (require == \cf8 'sharedmem' \cf5 and not \cf6 supports_sharedmem)\
    force_threads |= (\
        \cf5 not \cf6 explicit_backend \cf5 and \cf6 prefer == \cf8 'threads' \cf5 and not \cf6 uses_threads\
    )\
    \cf5 if \cf6 force_threads:\
        \cf4 # This backend does not match the shared memory constraint:\
        # fallback to the default thead-based backend.\
        \cf6 sharedmem_backend = BACKENDS[DEFAULT_THREAD_BACKEND](\
            \cf10 nesting_level\cf6 =nesting_level\
        )\
        \cf4 # Warn the user if we forced the backend to thread-based, while the\
        # user explicitely specified a non-thread-based backend.\
        \cf5 if \cf6 verbose >= \cf11 10 \cf5 and \cf6 explicit_backend:\
            \cf7 print\cf6 (\
                \cf8 f"Using \cf5 \{\cf6 sharedmem_backend.__class__.\cf12 __name__\cf5 \}\cf8  as "\
                f"joblib backend instead of \cf5 \{\cf6 backend.__class__.\cf12 __name__\cf5 \} \cf8 "\
                "as the latter does not provide shared memory semantics."\
            \cf6 )\
        \cf4 # Force to n_jobs=1 by default\
        \cf6 thread_config = backend_config.copy()\
        thread_config[\cf8 'n_jobs'\cf6 ] = \cf11 1\
        \cf5 return \cf6 sharedmem_backend, thread_config\
\
    \cf5 return \cf6 backend, backend_config\
\
\
\cf5 class \cf6 parallel_config:\
    
\f0\i \cf2 """Set the default backend or configuration for :class:`~joblib.Parallel`.\
\
    This is an alternative to directly passing keyword arguments to the\
    :class:`~joblib.Parallel` class constructor. It is particularly useful when\
    calling into library code that uses joblib internally but does not expose\
    the various parallel configuration arguments in its own API.\
\
    Parameters\
    ----------\
    backend : str or ParallelBackendBase instance, default=None\
        If ``backend`` is a string it must match a previously registered\
        implementation using the :func:`~register_parallel_backend` function.\
\
        By default the following backends are available:\
\
        - 'loky': single-host, process-based parallelism (used by default),\
        - 'threading': single-host, thread-based parallelism,\
        - 'multiprocessing': legacy single-host, process-based parallelism.\
\
        'loky' is recommended to run functions that manipulate Python objects.\
        'threading' is a low-overhead alternative that is most efficient for\
        functions that release the Global Interpreter Lock: e.g. I/O-bound\
        code or CPU-bound code in a few calls to native code that explicitly\
        releases the GIL. Note that on some rare systems (such as pyodide),\
        multiprocessing and loky may not be available, in which case joblib\
        defaults to threading.\
\
        In addition, if the ``dask`` and ``distributed`` Python packages are\
        installed, it is possible to use the 'dask' backend for better\
        scheduling of nested parallel calls without over-subscription and\
        potentially distribute parallel calls over a networked cluster of\
        several hosts.\
\
        It is also possible to use the distributed 'ray' backend for\
        distributing the workload to a cluster of nodes. See more details\
        in the Examples section below.\
\
        Alternatively the backend can be passed directly as an instance.\
\
    n_jobs : int, default=None\
        The maximum number of concurrently running jobs, such as the number\
        of Python worker processes when ``backend="loky"`` or the size of the\
        thread-pool when ``backend="threading"``.\
        If -1 all CPUs are used. If 1 is given, no parallel computing code\
        is used at all, which is useful for debugging. For ``n_jobs`` below -1,\
        (n_cpus + 1 + n_jobs) are used. Thus for ``n_jobs=-2``, all\
        CPUs but one are used.\
        ``None`` is a marker for 'unset' that will be interpreted as\
        ``n_jobs=1`` in most backends.\
\
    verbose : int, default=0\
        The verbosity level: if non zero, progress messages are\
        printed. Above 50, the output is sent to stdout.\
        The frequency of the messages increases with the verbosity level.\
        If it more than 10, all iterations are reported.\
\
    temp_folder : str, default=None\
        Folder to be used by the pool for memmapping large arrays\
        for sharing memory with worker processes. If None, this will try in\
        order:\
\
        - a folder pointed by the ``JOBLIB_TEMP_FOLDER`` environment\
          variable,\
        - ``/dev/shm`` if the folder exists and is writable: this is a\
          RAM disk filesystem available by default on modern Linux\
          distributions,\
        - the default system temporary folder that can be\
          overridden with ``TMP``, ``TMPDIR`` or ``TEMP`` environment\
          variables, typically ``/tmp`` under Unix operating systems.\
\
    max_nbytes int, str, or None, optional, default='1M'\
        Threshold on the size of arrays passed to the workers that\
        triggers automated memory mapping in temp_folder. Can be an int\
        in Bytes, or a human-readable string, e.g., '1M' for 1 megabyte.\
        Use None to disable memmapping of large arrays.\
\
    mmap_mode: \{None, 'r+', 'r', 'w+', 'c'\}, default='r'\
        Memmapping mode for numpy arrays passed to workers. None will\
        disable memmapping, other modes defined in the numpy.memmap doc:\
        https://numpy.org/doc/stable/reference/generated/numpy.memmap.html\
        Also, see 'max_nbytes' parameter documentation for more details.\
\
    prefer: str in \{'processes', 'threads'\} or None, default=None\
        Soft hint to choose the default backend.\
        The default process-based backend is 'loky' and the default\
        thread-based backend is 'threading'. Ignored if the ``backend``\
        parameter is specified.\
\
    require: 'sharedmem' or None, default=None\
        Hard constraint to select the backend. If set to 'sharedmem',\
        the selected backend will be single-host and thread-based.\
\
    inner_max_num_threads : int, default=None\
        If not None, overwrites the limit set on the number of threads\
        usable in some third-party library threadpools like OpenBLAS,\
        MKL or OpenMP. This is only used with the ``loky`` backend.\
\
    backend_params : dict\
        Additional parameters to pass to the backend constructor when\
        backend is a string.\
\
    Notes\
    -----\
    Joblib tries to limit the oversubscription by limiting the number of\
    threads usable in some third-party library threadpools like OpenBLAS, MKL\
    or OpenMP. The default limit in each worker is set to\
    ``max(cpu_count() // effective_n_jobs, 1)`` but this limit can be\
    overwritten with the ``inner_max_num_threads`` argument which will be used\
    to set this limit in the child processes.\
\
    .. versionadded:: 1.3\
\
    Examples\
    --------\
    >>> from operator import neg\
    >>> with parallel_config(backend='threading'):\
    ...     print(Parallel()(delayed(neg)(i + 1) for i in range(5)))\
    ...\
    [-1, -2, -3, -4, -5]\
\
    To use the 'ray' joblib backend add the following lines:\
\
    >>> from ray.util.joblib import register_ray  # doctest: +SKIP\
    >>> register_ray()  # doctest: +SKIP\
    >>> with parallel_config(backend="ray"):  # doctest: +SKIP\
    ...     print(Parallel()(delayed(neg)(i + 1) for i in range(5)))\
    [-1, -2, -3, -4, -5]\
\
    """\
    
\f1\i0 \cf5 def \cf12 __init__\cf6 (\
        \cf13 self\cf6 ,\
        backend=default_parallel_config[\cf8 "backend"\cf6 ],\
        *,\
        n_jobs=default_parallel_config[\cf8 "n_jobs"\cf6 ],\
        verbose=default_parallel_config[\cf8 "verbose"\cf6 ],\
        temp_folder=default_parallel_config[\cf8 "temp_folder"\cf6 ],\
        max_nbytes=default_parallel_config[\cf8 "max_nbytes"\cf6 ],\
        mmap_mode=default_parallel_config[\cf8 "mmap_mode"\cf6 ],\
        prefer=default_parallel_config[\cf8 "prefer"\cf6 ],\
        require=default_parallel_config[\cf8 "require"\cf6 ],\
        inner_max_num_threads=\cf5 None\cf6 ,\
        **backend_params\
    ):\
        \cf4 # Save the parallel info and set the active parallel config\
        \cf13 self\cf6 .old_parallel_config = \cf7 getattr\cf6 (\
            _backend, \cf8 "config"\cf6 , default_parallel_config\
        )\
\
        backend = \cf13 self\cf6 ._check_backend(\
            backend, inner_max_num_threads, **backend_params\
        )\
\
        new_config = \{\
            \cf8 "n_jobs"\cf6 : n_jobs,\
            \cf8 "verbose"\cf6 : verbose,\
            \cf8 "temp_folder"\cf6 : temp_folder,\
            \cf8 "max_nbytes"\cf6 : max_nbytes,\
            \cf8 "mmap_mode"\cf6 : mmap_mode,\
            \cf8 "prefer"\cf6 : prefer,\
            \cf8 "require"\cf6 : require,\
            \cf8 "backend"\cf6 : backend\
        \}\
        \cf13 self\cf6 .parallel_config = \cf13 self\cf6 .old_parallel_config.copy()\
        \cf13 self\cf6 .parallel_config.update(\{\
            k: v \cf5 for \cf6 k, v \cf5 in \cf6 new_config.items()\
            \cf5 if not \cf7 isinstance\cf6 (v, _Sentinel)\
        \})\
\
        \cf7 setattr\cf6 (_backend, \cf8 "config"\cf6 , \cf13 self\cf6 .parallel_config)\
\
    \cf5 def \cf9 _check_backend\cf6 (\cf13 self\cf6 , backend, inner_max_num_threads, **backend_params):\
        \cf5 if \cf6 backend \cf5 is \cf6 default_parallel_config[\cf8 'backend'\cf6 ]:\
            \cf5 if \cf6 inner_max_num_threads \cf5 is not None or \cf7 len\cf6 (backend_params) > \cf11 0\cf6 :\
                \cf5 raise \cf7 ValueError\cf6 (\
                    \cf8 "inner_max_num_threads and other constructor "\
                    "parameters backend_params are only supported "\
                    "when backend is not None."\
                \cf6 )\
            \cf5 return \cf6 backend\
\
        \cf5 if \cf7 isinstance\cf6 (backend, \cf7 str\cf6 ):\
            \cf4 # Handle non-registered or missing backends\
            \cf5 if \cf6 backend \cf5 not in \cf6 BACKENDS:\
                \cf5 if \cf6 backend \cf5 in \cf6 EXTERNAL_BACKENDS:\
                    register = EXTERNAL_BACKENDS[backend]\
                    register()\
                \cf5 elif \cf6 backend \cf5 in \cf6 MAYBE_AVAILABLE_BACKENDS:\
                    warnings.warn(\
                        \cf8 f"joblib backend '\cf5 \{\cf6 backend\cf5 \}\cf8 ' is not available on "\
                        f"your system, falling back to \cf5 \{\cf6 DEFAULT_BACKEND\cf5 \}\cf8 ."\cf6 ,\
                        \cf7 UserWarning\cf6 ,\
                        \cf10 stacklevel\cf6 =\cf11 2\
                    \cf6 )\
                    BACKENDS[backend] = BACKENDS[DEFAULT_BACKEND]\
                \cf5 else\cf6 :\
                    \cf5 raise \cf7 ValueError\cf6 (\
                        \cf8 f"Invalid backend: \cf5 \{\cf6 backend\cf5 \}\cf8 , expected one of "\
                        f"\cf5 \{\cf7 sorted\cf6 (BACKENDS.keys())\cf5 \}\cf8 "\
                    \cf6 )\
\
            backend = BACKENDS[backend](**backend_params)\
\
        \cf5 if \cf6 inner_max_num_threads \cf5 is not None\cf6 :\
            msg = (\
                \cf8 f"\cf5 \{\cf6 backend.__class__.\cf12 __name__\cf5 \}\cf8  does not accept setting the "\
                "inner_max_num_threads argument."\
            \cf6 )\
            \cf5 assert \cf6 backend.supports_inner_max_num_threads, msg\
            backend.inner_max_num_threads = inner_max_num_threads\
\
        \cf4 # If the nesting_level of the backend is not set previously, use the\
        # nesting level from the previous active_backend to set it\
        \cf5 if \cf6 backend.nesting_level \cf5 is None\cf6 :\
            parent_backend = \cf13 self\cf6 .old_parallel_config[\cf8 'backend'\cf6 ]\
            \cf5 if \cf6 parent_backend \cf5 is \cf6 default_parallel_config[\cf8 'backend'\cf6 ]:\
                nesting_level = \cf11 0\
            \cf5 else\cf6 :\
                nesting_level = parent_backend.nesting_level\
            backend.nesting_level = nesting_level\
\
        \cf5 return \cf6 backend\
\
    \cf5 def \cf12 __enter__\cf6 (\cf13 self\cf6 ):\
        \cf5 return \cf13 self\cf6 .parallel_config\
\
    \cf5 def \cf12 __exit__\cf6 (\cf13 self\cf6 , type, value, traceback):\
        \cf13 self\cf6 .unregister()\
\
    \cf5 def \cf9 unregister\cf6 (\cf13 self\cf6 ):\
        \cf7 setattr\cf6 (_backend, \cf8 "config"\cf6 , \cf13 self\cf6 .old_parallel_config)\
\
\
\cf5 class \cf6 parallel_backend(parallel_config):\
    
\f0\i \cf2 """Change the default backend used by Parallel inside a with block.\
\
    .. warning::\
        It is advised to use the :class:`~joblib.parallel_config` context\
        manager instead, which allows more fine-grained control over the\
        backend configuration.\
\
    If ``backend`` is a string it must match a previously registered\
    implementation using the :func:`~register_parallel_backend` function.\
\
    By default the following backends are available:\
\
    - 'loky': single-host, process-based parallelism (used by default),\
    - 'threading': single-host, thread-based parallelism,\
    - 'multiprocessing': legacy single-host, process-based parallelism.\
\
    'loky' is recommended to run functions that manipulate Python objects.\
    'threading' is a low-overhead alternative that is most efficient for\
    functions that release the Global Interpreter Lock: e.g. I/O-bound code or\
    CPU-bound code in a few calls to native code that explicitly releases the\
    GIL. Note that on some rare systems (such as Pyodide),\
    multiprocessing and loky may not be available, in which case joblib\
    defaults to threading.\
\
    You can also use the `Dask <https://docs.dask.org/en/stable/>`_ joblib\
    backend to distribute work across machines. This works well with\
    scikit-learn estimators with the ``n_jobs`` parameter, for example::\
\
    >>> import joblib  # doctest: +SKIP\
    >>> from sklearn.model_selection import GridSearchCV  # doctest: +SKIP\
    >>> from dask.distributed import Client, LocalCluster # doctest: +SKIP\
\
    >>> # create a local Dask cluster\
    >>> cluster = LocalCluster()  # doctest: +SKIP\
    >>> client = Client(cluster)  # doctest: +SKIP\
    >>> grid_search = GridSearchCV(estimator, param_grid, n_jobs=-1)\
    ... # doctest: +SKIP\
    >>> with joblib.parallel_backend("dask", scatter=[X, y]):  # doctest: +SKIP\
    ...     grid_search.fit(X, y)\
\
    It is also possible to use the distributed 'ray' backend for distributing\
    the workload to a cluster of nodes. To use the 'ray' joblib backend add\
    the following lines::\
\
     >>> from ray.util.joblib import register_ray  # doctest: +SKIP\
     >>> register_ray()  # doctest: +SKIP\
     >>> with parallel_backend("ray"):  # doctest: +SKIP\
     ...     print(Parallel()(delayed(neg)(i + 1) for i in range(5)))\
     [-1, -2, -3, -4, -5]\
\
    Alternatively the backend can be passed directly as an instance.\
\
    By default all available workers will be used (``n_jobs=-1``) unless the\
    caller passes an explicit value for the ``n_jobs`` parameter.\
\
    This is an alternative to passing a ``backend='backend_name'`` argument to\
    the :class:`~Parallel` class constructor. It is particularly useful when\
    calling into library code that uses joblib internally but does not expose\
    the backend argument in its own API.\
\
    >>> from operator import neg\
    >>> with parallel_backend('threading'):\
    ...     print(Parallel()(delayed(neg)(i + 1) for i in range(5)))\
    ...\
    [-1, -2, -3, -4, -5]\
\
    Joblib also tries to limit the oversubscription by limiting the number of\
    threads usable in some third-party library threadpools like OpenBLAS, MKL\
    or OpenMP. The default limit in each worker is set to\
    ``max(cpu_count() // effective_n_jobs, 1)`` but this limit can be\
    overwritten with the ``inner_max_num_threads`` argument which will be used\
    to set this limit in the child processes.\
\
    .. versionadded:: 0.10\
\
    See Also\
    --------\
    joblib.parallel_config : context manager to change the backend\
        configuration.\
    """\
    
\f1\i0 \cf5 def \cf12 __init__\cf6 (\cf13 self\cf6 , backend, n_jobs=-\cf11 1\cf6 , inner_max_num_threads=\cf5 None\cf6 ,\
                 **backend_params):\
\
        \cf7 super\cf6 ().\cf12 __init__\cf6 (\
            \cf10 backend\cf6 =backend,\
            \cf10 n_jobs\cf6 =n_jobs,\
            \cf10 inner_max_num_threads\cf6 =inner_max_num_threads,\
            **backend_params\
        )\
\
        \cf5 if \cf13 self\cf6 .old_parallel_config \cf5 is None\cf6 :\
            \cf13 self\cf6 .old_backend_and_jobs = \cf5 None\
        else\cf6 :\
            \cf13 self\cf6 .old_backend_and_jobs = (\
                \cf13 self\cf6 .old_parallel_config[\cf8 "backend"\cf6 ],\
                \cf13 self\cf6 .old_parallel_config[\cf8 "n_jobs"\cf6 ],\
            )\
        \cf13 self\cf6 .new_backend_and_jobs = (\
            \cf13 self\cf6 .parallel_config[\cf8 "backend"\cf6 ],\
            \cf13 self\cf6 .parallel_config[\cf8 "n_jobs"\cf6 ],\
        )\
\
    \cf5 def \cf12 __enter__\cf6 (\cf13 self\cf6 ):\
        \cf5 return \cf13 self\cf6 .new_backend_and_jobs\
\
\
\cf4 # Under Linux or OS X the default start method of multiprocessing\
# can cause third party libraries to crash. Under Python 3.4+ it is possible\
# to set an environment variable to switch the default start method from\
# 'fork' to 'forkserver' or 'spawn' to avoid this issue albeit at the cost\
# of causing semantic changes and some additional pool instantiation overhead.\
\cf6 DEFAULT_MP_CONTEXT = \cf5 None\
if \cf7 hasattr\cf6 (mp, \cf8 'get_context'\cf6 ):\
    method = os.environ.get(\cf8 'JOBLIB_START_METHOD'\cf6 , \cf8 ''\cf6 ).strip() \cf5 or None\
    if \cf6 method \cf5 is not None\cf6 :\
        DEFAULT_MP_CONTEXT = mp.get_context(\cf10 method\cf6 =method)\
\
\
\cf5 class \cf6 BatchedCalls(\cf7 object\cf6 ):\
    
\f0\i \cf2 """Wrap a sequence of (func, args, kwargs) tuples as a single callable"""\
\
    
\f1\i0 \cf5 def \cf12 __init__\cf6 (\cf13 self\cf6 , iterator_slice, backend_and_jobs, reducer_callback=\cf5 None\cf6 ,\
                 pickle_cache=\cf5 None\cf6 ):\
        \cf13 self\cf6 .items = \cf7 list\cf6 (iterator_slice)\
        \cf13 self\cf6 ._size = \cf7 len\cf6 (\cf13 self\cf6 .items)\
        \cf13 self\cf6 ._reducer_callback = reducer_callback\
        \cf5 if \cf7 isinstance\cf6 (backend_and_jobs, \cf7 tuple\cf6 ):\
            \cf13 self\cf6 ._backend, \cf13 self\cf6 ._n_jobs = backend_and_jobs\
        \cf5 else\cf6 :\
            \cf4 # this is for backward compatibility purposes. Before 0.12.6,\
            # nested backends were returned without n_jobs indications.\
            \cf13 self\cf6 ._backend, \cf13 self\cf6 ._n_jobs = backend_and_jobs, \cf5 None\
        \cf13 self\cf6 ._pickle_cache = pickle_cache \cf5 if \cf6 pickle_cache \cf5 is not None else \cf6 \{\}\
\
    \cf5 def \cf12 __call__\cf6 (\cf13 self\cf6 ):\
        \cf4 # Set the default nested backend to self._backend but do not set the\
        # change the default number of processes to -1\
        \cf5 with \cf6 parallel_config(\cf10 backend\cf6 =\cf13 self\cf6 ._backend, \cf10 n_jobs\cf6 =\cf13 self\cf6 ._n_jobs):\
            \cf5 return \cf6 [func(*args, **kwargs)\
                    \cf5 for \cf6 func, args, kwargs \cf5 in \cf13 self\cf6 .items]\
\
    \cf5 def \cf12 __reduce__\cf6 (\cf13 self\cf6 ):\
        \cf5 if \cf13 self\cf6 ._reducer_callback \cf5 is not None\cf6 :\
            \cf13 self\cf6 ._reducer_callback()\
        \cf4 # no need to pickle the callback.\
        \cf5 return \cf6 (\
            BatchedCalls,\
            (\cf13 self\cf6 .items, (\cf13 self\cf6 ._backend, \cf13 self\cf6 ._n_jobs), \cf5 None\cf6 ,\
             \cf13 self\cf6 ._pickle_cache)\
        )\
\
    \cf5 def \cf12 __len__\cf6 (\cf13 self\cf6 ):\
        \cf5 return \cf13 self\cf6 ._size\
\
\
\cf4 # Possible exit status for a task\
\cf6 TASK_DONE = \cf8 "Done"\
\cf6 TASK_ERROR = \cf8 "Error"\
\cf6 TASK_PENDING = \cf8 "Pending"\
\
\
\cf4 ###############################################################################\
# CPU count that works also when multiprocessing has been disabled via\
# the JOBLIB_MULTIPROCESSING environment variable\
\cf5 def \cf9 cpu_count\cf6 (only_physical_cores=\cf5 False\cf6 ):\
    
\f0\i \cf2 """Return the number of CPUs.\
\
    This delegates to loky.cpu_count that takes into account additional\
    constraints such as Linux CFS scheduler quotas (typically set by container\
    runtimes such as docker) and CPU affinity (for instance using the taskset\
    command on Linux).\
\
    If only_physical_cores is True, do not take hyperthreading / SMT logical\
    cores into account.\
    """\
    
\f1\i0 \cf5 if \cf6 mp \cf5 is None\cf6 :\
        \cf5 return \cf11 1\
\
    \cf5 return \cf6 loky.cpu_count(\cf10 only_physical_cores\cf6 =only_physical_cores)\
\
\
\cf4 ###############################################################################\
# For verbosity\
\
\cf5 def \cf9 _verbosity_filter\cf6 (index, verbose):\
    
\f0\i \cf2 """ Returns False for indices increasingly apart, the distance\
        depending on the value of verbose.\
\
        We use a lag increasing as the square of index\
    """\
    
\f1\i0 \cf5 if not \cf6 verbose:\
        \cf5 return True\
    elif \cf6 verbose > \cf11 10\cf6 :\
        \cf5 return False\
    if \cf6 index == \cf11 0\cf6 :\
        \cf5 return False\
    \cf6 verbose = \cf11 .5 \cf6 * (\cf11 11 \cf6 - verbose) ** \cf11 2\
    \cf6 scale = sqrt(index / verbose)\
    next_scale = sqrt((index + \cf11 1\cf6 ) / verbose)\
    \cf5 return \cf6 (\cf7 int\cf6 (next_scale) == \cf7 int\cf6 (scale))\
\
\
\cf4 ###############################################################################\
\cf5 def \cf9 delayed\cf6 (function):\
    
\f0\i \cf2 """Decorator used to capture the arguments of a function."""\
\
    
\f1\i0 \cf5 def \cf9 delayed_function\cf6 (*args, **kwargs):\
        \cf5 return \cf6 function, args, kwargs\
    \cf5 try\cf6 :\
        delayed_function = functools.wraps(function)(delayed_function)\
    \cf5 except \cf7 AttributeError\cf6 :\
        \cf8 " functools.wraps fails on some callable objects "\
    \cf5 return \cf6 delayed_function\
\
\
\cf4 ###############################################################################\
\cf5 class \cf6 BatchCompletionCallBack(\cf7 object\cf6 ):\
    
\f0\i \cf2 """Callback to keep track of completed results and schedule the next tasks.\
\
    This callable is executed by the parent process whenever a worker process\
    has completed a batch of tasks.\
\
    It is used for progress reporting, to update estimate of the batch\
    processing duration and to schedule the next batch of tasks to be\
    processed.\
\
    It is assumed that this callback will always be triggered by the backend\
    right after the end of a task, in case of success as well as in case of\
    failure.\
    """\
\
    
\f1\i0 \cf4 ##########################################################################\
    #                   METHODS CALLED BY THE MAIN THREAD                    #\
    ##########################################################################\
    \cf5 def \cf12 __init__\cf6 (\cf13 self\cf6 , dispatch_timestamp, batch_size, parallel):\
        \cf13 self\cf6 .dispatch_timestamp = dispatch_timestamp\
        \cf13 self\cf6 .batch_size = batch_size\
        \cf13 self\cf6 .parallel = parallel\
        \cf13 self\cf6 .parallel_call_id = parallel._call_id\
\
        \cf4 # Internals to keep track of the status and outcome of the task.\
\
        # Used to hold a reference to the future-like object returned by the\
        # backend after launching this task\
        # This will be set later when calling `register_job`, as it is only\
        # created once the task has been submitted.\
        \cf13 self\cf6 .job = \cf5 None\
\
        if not \cf6 parallel._backend.supports_retrieve_callback:\
            \cf4 # The status is only used for asynchronous result retrieval in the\
            # callback.\
            \cf13 self\cf6 .status = \cf5 None\
        else\cf6 :\
            \cf4 # The initial status for the job is TASK_PENDING.\
            # Once it is done, it will be either TASK_DONE, or TASK_ERROR.\
            \cf13 self\cf6 .status = TASK_PENDING\
\
    \cf5 def \cf9 register_job\cf6 (\cf13 self\cf6 , job):\
        
\f0\i \cf2 """Register the object returned by `apply_async`."""\
        
\f1\i0 \cf13 self\cf6 .job = job\
\
    \cf5 def \cf9 get_result\cf6 (\cf13 self\cf6 , timeout):\
        
\f0\i \cf2 """Returns the raw result of the task that was submitted.\
\
        If the task raised an exception rather than returning, this same\
        exception will be raised instead.\
\
        If the backend supports the retrieval callback, it is assumed that this\
        method is only called after the result has been registered. It is\
        ensured by checking that `self.status(timeout)` does not return\
        TASK_PENDING. In this case, `get_result` directly returns the\
        registered result (or raise the registered exception).\
\
        For other backends, there are no such assumptions, but `get_result`\
        still needs to synchronously retrieve the result before it can\
        return it or raise. It will block at most `self.timeout` seconds\
        waiting for retrieval to complete, after that it raises a TimeoutError.\
        """\
\
        
\f1\i0 \cf6 backend = \cf13 self\cf6 .parallel._backend\
\
        \cf5 if \cf6 backend.supports_retrieve_callback:\
            \cf4 # We assume that the result has already been retrieved by the\
            # callback thread, and is stored internally. It's just waiting to\
            # be returned.\
            \cf5 return \cf13 self\cf6 ._return_or_raise()\
\
        \cf4 # For other backends, the main thread needs to run the retrieval step.\
        \cf5 try\cf6 :\
            \cf5 if \cf6 backend.supports_timeout:\
                result = \cf13 self\cf6 .job.get(\cf10 timeout\cf6 =timeout)\
            \cf5 else\cf6 :\
                result = \cf13 self\cf6 .job.get()\
            outcome = \cf7 dict\cf6 (\cf10 result\cf6 =result, \cf10 status\cf6 =TASK_DONE)\
        \cf5 except \cf7 BaseException \cf5 as \cf6 e:\
            outcome = \cf7 dict\cf6 (\cf10 result\cf6 =e, \cf10 status\cf6 =TASK_ERROR)\
        \cf13 self\cf6 ._register_outcome(outcome)\
\
        \cf5 return \cf13 self\cf6 ._return_or_raise()\
\
    \cf5 def \cf9 _return_or_raise\cf6 (\cf13 self\cf6 ):\
        \cf5 try\cf6 :\
            \cf5 if \cf13 self\cf6 .status == TASK_ERROR:\
                \cf5 raise \cf13 self\cf6 ._result\
            \cf5 return \cf13 self\cf6 ._result\
        \cf5 finally\cf6 :\
            \cf5 del \cf13 self\cf6 ._result\
\
    \cf5 def \cf9 get_status\cf6 (\cf13 self\cf6 , timeout):\
        
\f0\i \cf2 """Get the status of the task.\
\
        This function also checks if the timeout has been reached and register\
        the TimeoutError outcome when it is the case.\
        """\
        
\f1\i0 \cf5 if \cf6 timeout \cf5 is None or \cf13 self\cf6 .status != TASK_PENDING:\
            \cf5 return \cf13 self\cf6 .status\
\
        \cf4 # The computation are running and the status is pending.\
        # Check that we did not wait for this jobs more than `timeout`.\
        \cf6 now = time.time()\
        \cf5 if not \cf7 hasattr\cf6 (\cf13 self\cf6 , \cf8 "_completion_timeout_counter"\cf6 ):\
            \cf13 self\cf6 ._completion_timeout_counter = now\
\
        \cf5 if \cf6 (now - \cf13 self\cf6 ._completion_timeout_counter) > timeout:\
            outcome = \cf7 dict\cf6 (\cf10 result\cf6 =TimeoutError(), \cf10 status\cf6 =TASK_ERROR)\
            \cf13 self\cf6 ._register_outcome(outcome)\
\
        \cf5 return \cf13 self\cf6 .status\
\
    \cf4 ##########################################################################\
    #                     METHODS CALLED BY CALLBACK THREADS                 #\
    ##########################################################################\
    \cf5 def \cf12 __call__\cf6 (\cf13 self\cf6 , out):\
        
\f0\i \cf2 """Function called by the callback thread after a job is completed."""\
\
        
\f1\i0 \cf4 # If the backend doesn't support callback retrievals, the next batch of\
        # tasks is dispatched regardless. The result will be retrieved by the\
        # main thread when calling `get_result`.\
        \cf5 if not \cf13 self\cf6 .parallel._backend.supports_retrieve_callback:\
            \cf13 self\cf6 ._dispatch_new()\
            \cf5 return\
\
        \cf4 # If the backend supports retrieving the result in the callback, it\
        # registers the task outcome (TASK_ERROR or TASK_DONE), and schedules\
        # the next batch if needed.\
        \cf5 with \cf13 self\cf6 .parallel._lock:\
            \cf4 # Edge case where while the task was processing, the `parallel`\
            # instance has been reset and a new call has been issued, but the\
            # worker managed to complete the task and trigger this callback\
            # call just before being aborted by the reset.\
            \cf5 if \cf13 self\cf6 .parallel._call_id != \cf13 self\cf6 .parallel_call_id:\
                \cf5 return\
\
            \cf4 # When aborting, stop as fast as possible and do not retrieve the\
            # result as it won't be returned by the Parallel call.\
            \cf5 if \cf13 self\cf6 .parallel._aborting:\
                \cf5 return\
\
            \cf4 # Retrieves the result of the task in the main process and dispatch\
            # a new batch if needed.\
            \cf6 job_succeeded = \cf13 self\cf6 ._retrieve_result(out)\
\
        \cf5 if \cf6 job_succeeded:\
            \cf13 self\cf6 ._dispatch_new()\
\
    \cf5 def \cf9 _dispatch_new\cf6 (\cf13 self\cf6 ):\
        
\f0\i \cf2 """Schedule the next batch of tasks to be processed."""\
\
        
\f1\i0 \cf4 # This steps ensure that auto-baching works as expected.\
        \cf6 this_batch_duration = time.time() - \cf13 self\cf6 .dispatch_timestamp\
        \cf13 self\cf6 .parallel._backend.batch_completed(\cf13 self\cf6 .batch_size,\
                                               this_batch_duration)\
\
        \cf4 # Schedule the next batch of tasks.\
        \cf5 with \cf13 self\cf6 .parallel._lock:\
            \cf13 self\cf6 .parallel.n_completed_tasks += \cf13 self\cf6 .batch_size\
            \cf13 self\cf6 .parallel.print_progress()\
            \cf5 if \cf13 self\cf6 .parallel._original_iterator \cf5 is not None\cf6 :\
                \cf13 self\cf6 .parallel.dispatch_next()\
\
    \cf5 def \cf9 _retrieve_result\cf6 (\cf13 self\cf6 , out):\
        
\f0\i \cf2 """Fetch and register the outcome of a task.\
\
        Return True if the task succeeded, False otherwise.\
        This function is only called by backends that support retrieving\
        the task result in the callback thread.\
        """\
        
\f1\i0 \cf5 try\cf6 :\
            result = \cf13 self\cf6 .parallel._backend.retrieve_result_callback(out)\
            outcome = \cf7 dict\cf6 (\cf10 status\cf6 =TASK_DONE, \cf10 result\cf6 =result)\
        \cf5 except \cf7 BaseException \cf5 as \cf6 e:\
            \cf4 # Avoid keeping references to parallel in the error.\
            \cf6 e.__traceback__ = \cf5 None\
            \cf6 outcome = \cf7 dict\cf6 (\cf10 result\cf6 =e, \cf10 status\cf6 =TASK_ERROR)\
\
        \cf13 self\cf6 ._register_outcome(outcome)\
        \cf5 return \cf6 outcome[\cf8 'status'\cf6 ] != TASK_ERROR\
\
    \cf4 ##########################################################################\
    #            This method can be called either in the main thread         #\
    #                        or in the callback thread.                      #\
    ##########################################################################\
    \cf5 def \cf9 _register_outcome\cf6 (\cf13 self\cf6 , outcome):\
        
\f0\i \cf2 """Register the outcome of a task.\
\
        This method can be called only once, future calls will be ignored.\
        """\
        
\f1\i0 \cf4 # Covers the edge case where the main thread tries to register a\
        # `TimeoutError` while the callback thread tries to register a result\
        # at the same time.\
        \cf5 with \cf13 self\cf6 .parallel._lock:\
            \cf5 if \cf13 self\cf6 .status \cf5 not in \cf6 (TASK_PENDING, \cf5 None\cf6 ):\
                \cf5 return\
            \cf13 self\cf6 .status = outcome[\cf8 "status"\cf6 ]\
\
        \cf13 self\cf6 ._result = outcome[\cf8 "result"\cf6 ]\
\
        \cf4 # Once the result and the status are extracted, the last reference to\
        # the job can be deleted.\
        \cf13 self\cf6 .job = \cf5 None\
\
        \cf4 # As soon as an error as been spotted, early stopping flags are sent to\
        # the `parallel` instance.\
        \cf5 if \cf13 self\cf6 .status == TASK_ERROR:\
            \cf13 self\cf6 .parallel._exception = \cf5 True\
            \cf13 self\cf6 .parallel._aborting = \cf5 True\
\
\
\cf4 ###############################################################################\
\cf5 def \cf9 register_parallel_backend\cf6 (name, factory, make_default=\cf5 False\cf6 ):\
    
\f0\i \cf2 """Register a new Parallel backend factory.\
\
    The new backend can then be selected by passing its name as the backend\
    argument to the :class:`~Parallel` class. Moreover, the default backend can\
    be overwritten globally by setting make_default=True.\
\
    The factory can be any callable that takes no argument and return an\
    instance of ``ParallelBackendBase``.\
\
    Warning: this function is experimental and subject to change in a future\
    version of joblib.\
\
    .. versionadded:: 0.10\
    """\
    
\f1\i0 \cf6 BACKENDS[name] = factory\
    \cf5 if \cf6 make_default:\
        \cf5 global \cf6 DEFAULT_BACKEND\
        DEFAULT_BACKEND = name\
\
\
\cf5 def \cf9 effective_n_jobs\cf6 (n_jobs=-\cf11 1\cf6 ):\
    
\f0\i \cf2 """Determine the number of jobs that can actually run in parallel\
\
    n_jobs is the number of workers requested by the callers. Passing n_jobs=-1\
    means requesting all available workers for instance matching the number of\
    CPU cores on the worker host(s).\
\
    This method should return a guesstimate of the number of workers that can\
    actually perform work concurrently with the currently enabled default\
    backend. The primary use case is to make it possible for the caller to know\
    in how many chunks to slice the work.\
\
    In general working on larger data chunks is more efficient (less scheduling\
    overhead and better use of CPU cache prefetching heuristics) as long as all\
    the workers have enough work to do.\
\
    Warning: this function is experimental and subject to change in a future\
    version of joblib.\
\
    .. versionadded:: 0.10\
    """\
    
\f1\i0 \cf5 if \cf6 n_jobs == \cf11 1\cf6 :\
        \cf5 return \cf11 1\
\
    \cf6 backend, backend_n_jobs = get_active_backend()\
    \cf5 if \cf6 n_jobs \cf5 is None\cf6 :\
        n_jobs = backend_n_jobs\
    \cf5 return \cf6 backend.effective_n_jobs(\cf10 n_jobs\cf6 =n_jobs)\
\
\
\cf4 ###############################################################################\
\cf5 class \cf6 Parallel(Logger):\
    
\f0\i \cf2 ''' Helper class for readable parallel mapping.\
\
        Read more in the :ref:`User Guide <parallel>`.\
\
        Parameters\
        ----------\
        n_jobs: int, default: None\
            The maximum number of concurrently running jobs, such as the number\
            of Python worker processes when backend="multiprocessing"\
            or the size of the thread-pool when backend="threading".\
            If -1 all CPUs are used.\
            If 1 is given, no parallel computing code is used at all, and the\
            behavior amounts to a simple python `for` loop. This mode is not\
            compatible with `timeout`.\
            For n_jobs below -1, (n_cpus + 1 + n_jobs) are used. Thus for\
            n_jobs = -2, all CPUs but one are used.\
            None is a marker for 'unset' that will be interpreted as n_jobs=1\
            unless the call is performed under a :func:`~parallel_config`\
            context manager that sets another value for ``n_jobs``.\
        backend: str, ParallelBackendBase instance or None, default: 'loky'\
            Specify the parallelization backend implementation.\
            Supported backends are:\
\
            - "loky" used by default, can induce some\
              communication and memory overhead when exchanging input and\
              output data with the worker Python processes. On some rare\
              systems (such as Pyiodide), the loky backend may not be\
              available.\
            - "multiprocessing" previous process-based backend based on\
              `multiprocessing.Pool`. Less robust than `loky`.\
            - "threading" is a very low-overhead backend but it suffers\
              from the Python Global Interpreter Lock if the called function\
              relies a lot on Python objects. "threading" is mostly useful\
              when the execution bottleneck is a compiled extension that\
              explicitly releases the GIL (for instance a Cython loop wrapped\
              in a "with nogil" block or an expensive call to a library such\
              as NumPy).\
            - finally, you can register backends by calling\
              :func:`~register_parallel_backend`. This will allow you to\
              implement a backend of your liking.\
\
            It is not recommended to hard-code the backend name in a call to\
            :class:`~Parallel` in a library. Instead it is recommended to set\
            soft hints (prefer) or hard constraints (require) so as to make it\
            possible for library users to change the backend from the outside\
            using the :func:`~parallel_config` context manager.\
        return_as: str in \{'list', 'generator'\}, default: 'list'\
            If 'list', calls to this instance will return a list, only when\
            all results have been processed and retrieved.\
            If 'generator', it will return a generator that yields the results\
            as soon as they are available, in the order the tasks have been\
            submitted with.\
            Future releases are planned to also support 'generator_unordered',\
            in which case the generator immediately yields available results\
            independently of the submission order.\
        prefer: str in \{'processes', 'threads'\} or None, default: None\
            Soft hint to choose the default backend if no specific backend\
            was selected with the :func:`~parallel_config` context manager.\
            The default process-based backend is 'loky' and the default\
            thread-based backend is 'threading'. Ignored if the ``backend``\
            parameter is specified.\
        require: 'sharedmem' or None, default None\
            Hard constraint to select the backend. If set to 'sharedmem',\
            the selected backend will be single-host and thread-based even\
            if the user asked for a non-thread based backend with\
            :func:`~joblib.parallel_config`.\
        verbose: int, optional\
            The verbosity level: if non zero, progress messages are\
            printed. Above 50, the output is sent to stdout.\
            The frequency of the messages increases with the verbosity level.\
            If it more than 10, all iterations are reported.\
        timeout: float, optional\
            Timeout limit for each task to complete.  If any task takes longer\
            a TimeOutError will be raised. Only applied when n_jobs != 1\
        pre_dispatch: \{'all', integer, or expression, as in '3*n_jobs'\}\
            The number of batches (of tasks) to be pre-dispatched.\
            Default is '2*n_jobs'. When batch_size="auto" this is reasonable\
            default and the workers should never starve. Note that only basic\
            arithmetics are allowed here and no modules can be used in this\
            expression.\
        batch_size: int or 'auto', default: 'auto'\
            The number of atomic tasks to dispatch at once to each\
            worker. When individual evaluations are very fast, dispatching\
            calls to workers can be slower than sequential computation because\
            of the overhead. Batching fast computations together can mitigate\
            this.\
            The ``'auto'`` strategy keeps track of the time it takes for a\
            batch to complete, and dynamically adjusts the batch size to keep\
            the time on the order of half a second, using a heuristic. The\
            initial batch size is 1.\
            ``batch_size="auto"`` with ``backend="threading"`` will dispatch\
            batches of a single task at a time as the threading backend has\
            very little overhead and using larger batch size has not proved to\
            bring any gain in that case.\
        temp_folder: str, optional\
            Folder to be used by the pool for memmapping large arrays\
            for sharing memory with worker processes. If None, this will try in\
            order:\
\
            - a folder pointed by the JOBLIB_TEMP_FOLDER environment\
              variable,\
            - /dev/shm if the folder exists and is writable: this is a\
              RAM disk filesystem available by default on modern Linux\
              distributions,\
            - the default system temporary folder that can be\
              overridden with TMP, TMPDIR or TEMP environment\
              variables, typically /tmp under Unix operating systems.\
\
            Only active when backend="loky" or "multiprocessing".\
        max_nbytes int, str, or None, optional, 1M by default\
            Threshold on the size of arrays passed to the workers that\
            triggers automated memory mapping in temp_folder. Can be an int\
            in Bytes, or a human-readable string, e.g., '1M' for 1 megabyte.\
            Use None to disable memmapping of large arrays.\
            Only active when backend="loky" or "multiprocessing".\
        mmap_mode: \{None, 'r+', 'r', 'w+', 'c'\}, default: 'r'\
            Memmapping mode for numpy arrays passed to workers. None will\
            disable memmapping, other modes defined in the numpy.memmap doc:\
            https://numpy.org/doc/stable/reference/generated/numpy.memmap.html\
            Also, see 'max_nbytes' parameter documentation for more details.\
\
        Notes\
        -----\
\
        This object uses workers to compute in parallel the application of a\
        function to many different arguments. The main functionality it brings\
        in addition to using the raw multiprocessing or concurrent.futures API\
        are (see examples for details):\
\
        * More readable code, in particular since it avoids\
          constructing list of arguments.\
\
        * Easier debugging:\
            - informative tracebacks even when the error happens on\
              the client side\
            - using 'n_jobs=1' enables to turn off parallel computing\
              for debugging without changing the codepath\
            - early capture of pickling errors\
\
        * An optional progress meter.\
\
        * Interruption of multiprocesses jobs with 'Ctrl-C'\
\
        * Flexible pickling control for the communication to and from\
          the worker processes.\
\
        * Ability to use shared memory efficiently with worker\
          processes for large numpy-based datastructures.\
\
        Note that the intended usage is to run one call at a time. Multiple\
        calls to the same Parallel object will result in a ``RuntimeError``\
\
        Examples\
        --------\
\
        A simple example:\
\
        >>> from math import sqrt\
        >>> from joblib import Parallel, delayed\
        >>> Parallel(n_jobs=1)(delayed(sqrt)(i**2) for i in range(10))\
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]\
\
        Reshaping the output when the function has several return\
        values:\
\
        >>> from math import modf\
        >>> from joblib import Parallel, delayed\
        >>> r = Parallel(n_jobs=1)(delayed(modf)(i/2.) for i in range(10))\
        >>> res, i = zip(*r)\
        >>> res\
        (0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5)\
        >>> i\
        (0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0)\
\
        The progress meter: the higher the value of `verbose`, the more\
        messages:\
\
        >>> from time import sleep\
        >>> from joblib import Parallel, delayed\
        >>> r = Parallel(n_jobs=2, verbose=10)(\
        ...     delayed(sleep)(.2) for _ in range(10)) #doctest: +SKIP\
        [Parallel(n_jobs=2)]: Done   1 tasks      | elapsed:    0.6s\
        [Parallel(n_jobs=2)]: Done   4 tasks      | elapsed:    0.8s\
        [Parallel(n_jobs=2)]: Done  10 out of  10 | elapsed:    1.4s finished\
\
        Traceback example, note how the line of the error is indicated\
        as well as the values of the parameter passed to the function that\
        triggered the exception, even though the traceback happens in the\
        child process:\
\
        >>> from heapq import nlargest\
        >>> from joblib import Parallel, delayed\
        >>> Parallel(n_jobs=2)(\
        ... delayed(nlargest)(2, n) for n in (range(4), 'abcde', 3))\
        ... # doctest: +SKIP\
        -----------------------------------------------------------------------\
        Sub-process traceback:\
        -----------------------------------------------------------------------\
        TypeError                                      Mon Nov 12 11:37:46 2012\
        PID: 12934                                Python 2.7.3: /usr/bin/python\
        ........................................................................\
        /usr/lib/python2.7/heapq.pyc in nlargest(n=2, iterable=3, key=None)\
            419         if n >= size:\
            420             return sorted(iterable, key=key, reverse=True)[:n]\
            421\
            422     # When key is none, use simpler decoration\
            423     if key is None:\
        --> 424         it = izip(iterable, count(0,-1))           # decorate\
            425         result = _nlargest(n, it)\
            426         return map(itemgetter(0), result)          # undecorate\
            427\
            428     # General case, slowest method\
         TypeError: izip argument #1 must support iteration\
        _______________________________________________________________________\
\
\
        Using pre_dispatch in a producer/consumer situation, where the\
        data is generated on the fly. Note how the producer is first\
        called 3 times before the parallel loop is initiated, and then\
        called to generate new data on the fly:\
\
        >>> from math import sqrt\
        >>> from joblib import Parallel, delayed\
        >>> def producer():\
        ...     for i in range(6):\
        ...         print('Produced %s' % i)\
        ...         yield i\
        >>> out = Parallel(n_jobs=2, verbose=100, pre_dispatch='1.5*n_jobs')(\
        ...     delayed(sqrt)(i) for i in producer()) #doctest: +SKIP\
        Produced 0\
        Produced 1\
        Produced 2\
        [Parallel(n_jobs=2)]: Done 1 jobs     | elapsed:  0.0s\
        Produced 3\
        [Parallel(n_jobs=2)]: Done 2 jobs     | elapsed:  0.0s\
        Produced 4\
        [Parallel(n_jobs=2)]: Done 3 jobs     | elapsed:  0.0s\
        Produced 5\
        [Parallel(n_jobs=2)]: Done 4 jobs     | elapsed:  0.0s\
        [Parallel(n_jobs=2)]: Done 6 out of 6 | elapsed:  0.0s remaining: 0.0s\
        [Parallel(n_jobs=2)]: Done 6 out of 6 | elapsed:  0.0s finished\
\
    '''\
    
\f1\i0 \cf5 def \cf12 __init__\cf6 (\
        \cf13 self\cf6 ,\
        n_jobs=default_parallel_config[\cf8 "n_jobs"\cf6 ],\
        backend=default_parallel_config[\cf8 'backend'\cf6 ],\
        return_as=\cf8 "list"\cf6 ,\
        verbose=default_parallel_config[\cf8 "verbose"\cf6 ],\
        timeout=\cf5 None\cf6 ,\
        pre_dispatch=\cf8 '2 * n_jobs'\cf6 ,\
        batch_size=\cf8 'auto'\cf6 ,\
        temp_folder=default_parallel_config[\cf8 "temp_folder"\cf6 ],\
        max_nbytes=default_parallel_config[\cf8 "max_nbytes"\cf6 ],\
        mmap_mode=default_parallel_config[\cf8 "mmap_mode"\cf6 ],\
        prefer=default_parallel_config[\cf8 "prefer"\cf6 ],\
        require=default_parallel_config[\cf8 "require"\cf6 ],\
    ):\
        \cf4 # Initiate parent Logger class state\
        \cf7 super\cf6 ().\cf12 __init__\cf6 ()\
\
        \cf4 # Interpret n_jobs=None as 'unset'\
        \cf5 if \cf6 n_jobs \cf5 is None\cf6 :\
            n_jobs = default_parallel_config[\cf8 "n_jobs"\cf6 ]\
\
        active_backend, context_config = _get_active_backend(\
            \cf10 prefer\cf6 =prefer, \cf10 require\cf6 =require, \cf10 verbose\cf6 =verbose\
        )\
\
        nesting_level = active_backend.nesting_level\
\
        \cf13 self\cf6 .verbose = _get_config_param(verbose, context_config, \cf8 "verbose"\cf6 )\
        \cf13 self\cf6 .timeout = timeout\
        \cf13 self\cf6 .pre_dispatch = pre_dispatch\
\
        \cf5 if \cf6 return_as \cf5 not in \cf6 \{\cf8 "list"\cf6 , \cf8 "generator"\cf6 \}:\
            \cf5 raise \cf7 ValueError\cf6 (\
                \cf8 'Expected `return_as` parameter to be a string equal to "list"'\
                f' or "generator", but got \cf5 \{\cf6 return_as\cf5 \}\cf8  instead'\
            \cf6 )\
        \cf13 self\cf6 .return_as = return_as\
        \cf13 self\cf6 .return_generator = return_as != \cf8 "list"\
\
        \cf4 # Check if we are under a parallel_config or parallel_backend\
        # context manager and use the config from the context manager\
        # for arguments that are not explicitly set.\
        \cf13 self\cf6 ._backend_args = \{\
            k: _get_config_param(param, context_config, k) \cf5 for \cf6 param, k \cf5 in \cf6 [\
                (max_nbytes, \cf8 "max_nbytes"\cf6 ),\
                (temp_folder, \cf8 "temp_folder"\cf6 ),\
                (mmap_mode, \cf8 "mmap_mode"\cf6 ),\
                (prefer, \cf8 "prefer"\cf6 ),\
                (require, \cf8 "require"\cf6 ),\
                (verbose, \cf8 "verbose"\cf6 ),\
            ]\
        \}\
\
        \cf5 if \cf7 isinstance\cf6 (\cf13 self\cf6 ._backend_args[\cf8 "max_nbytes"\cf6 ], \cf7 str\cf6 ):\
            \cf13 self\cf6 ._backend_args[\cf8 "max_nbytes"\cf6 ] = memstr_to_bytes(\
                \cf13 self\cf6 ._backend_args[\cf8 "max_nbytes"\cf6 ]\
            )\
        \cf13 self\cf6 ._backend_args[\cf8 "verbose"\cf6 ] = \cf7 max\cf6 (\
            \cf11 0\cf6 , \cf13 self\cf6 ._backend_args[\cf8 "verbose"\cf6 ] - \cf11 50\
        \cf6 )\
\
        \cf5 if \cf6 DEFAULT_MP_CONTEXT \cf5 is not None\cf6 :\
            \cf13 self\cf6 ._backend_args[\cf8 'context'\cf6 ] = DEFAULT_MP_CONTEXT\
        \cf5 elif \cf7 hasattr\cf6 (mp, \cf8 "get_context"\cf6 ):\
            \cf13 self\cf6 ._backend_args[\cf8 'context'\cf6 ] = mp.get_context()\
\
        \cf5 if \cf6 backend \cf5 is \cf6 default_parallel_config[\cf8 'backend'\cf6 ] \cf5 or \cf6 backend \cf5 is None\cf6 :\
            backend = active_backend\
\
        \cf5 elif \cf7 isinstance\cf6 (backend, ParallelBackendBase):\
            \cf4 # Use provided backend as is, with the current nesting_level if it\
            # is not set yet.\
            \cf5 if \cf6 backend.nesting_level \cf5 is None\cf6 :\
                backend.nesting_level = nesting_level\
\
        \cf5 elif \cf7 hasattr\cf6 (backend, \cf8 'Pool'\cf6 ) \cf5 and \cf7 hasattr\cf6 (backend, \cf8 'Lock'\cf6 ):\
            \cf4 # Make it possible to pass a custom multiprocessing context as\
            # backend to change the start method to forkserver or spawn or\
            # preload modules on the forkserver helper process.\
            \cf13 self\cf6 ._backend_args[\cf8 'context'\cf6 ] = backend\
            backend = MultiprocessingBackend(\cf10 nesting_level\cf6 =nesting_level)\
\
        \cf5 elif \cf6 backend \cf5 not in \cf6 BACKENDS \cf5 and \cf6 backend \cf5 in \cf6 MAYBE_AVAILABLE_BACKENDS:\
            warnings.warn(\
                \cf8 f"joblib backend '\cf5 \{\cf6 backend\cf5 \}\cf8 ' is not available on "\
                f"your system, falling back to \cf5 \{\cf6 DEFAULT_BACKEND\cf5 \}\cf8 ."\cf6 ,\
                \cf7 UserWarning\cf6 ,\
                \cf10 stacklevel\cf6 =\cf11 2\cf6 )\
            BACKENDS[backend] = BACKENDS[DEFAULT_BACKEND]\
            backend = BACKENDS[DEFAULT_BACKEND](\cf10 nesting_level\cf6 =nesting_level)\
\
        \cf5 else\cf6 :\
            \cf5 try\cf6 :\
                backend_factory = BACKENDS[backend]\
            \cf5 except \cf7 KeyError \cf5 as \cf6 e:\
                \cf5 raise \cf7 ValueError\cf6 (\cf8 "Invalid backend: %s, expected one of %r"\
                                 \cf6 % (backend, \cf7 sorted\cf6 (BACKENDS.keys()))) \cf5 from \cf6 e\
            backend = backend_factory(\cf10 nesting_level\cf6 =nesting_level)\
\
        n_jobs = _get_config_param(n_jobs, context_config, \cf8 "n_jobs"\cf6 )\
        \cf5 if \cf6 n_jobs \cf5 is None\cf6 :\
            \cf4 # No specific context override and no specific value request:\
            # default to the default of the backend.\
            \cf6 n_jobs = backend.default_n_jobs\
        \cf13 self\cf6 .n_jobs = n_jobs\
\
        \cf5 if \cf6 (require == \cf8 'sharedmem' \cf5 and\
                not \cf7 getattr\cf6 (backend, \cf8 'supports_sharedmem'\cf6 , \cf5 False\cf6 )):\
            \cf5 raise \cf7 ValueError\cf6 (\cf8 "Backend %s does not support shared memory"\
                             \cf6 % backend)\
\
        \cf5 if \cf6 (batch_size == \cf8 'auto' \cf5 or \cf7 isinstance\cf6 (batch_size, Integral) \cf5 and\
                \cf6 batch_size > \cf11 0\cf6 ):\
            \cf13 self\cf6 .batch_size = batch_size\
        \cf5 else\cf6 :\
            \cf5 raise \cf7 ValueError\cf6 (\
                \cf8 "batch_size must be 'auto' or a positive integer, got: %r"\
                \cf6 % batch_size)\
\
        \cf5 if not \cf7 isinstance\cf6 (backend, SequentialBackend):\
            \cf5 if \cf13 self\cf6 .return_generator \cf5 and not \cf6 backend.supports_return_generator:\
                \cf5 raise \cf7 ValueError\cf6 (\
                    \cf8 "Backend \{\} does not support "\
                    "return_as=\{\}"\cf6 .format(backend, return_as)\
                )\
            \cf4 # This lock is used to coordinate the main thread of this process\
            # with the async callback thread of our the pool.\
            \cf13 self\cf6 ._lock = threading.RLock()\
            \cf13 self\cf6 ._jobs = collections.deque()\
            \cf13 self\cf6 ._pending_outputs = \cf7 list\cf6 ()\
            \cf13 self\cf6 ._ready_batches = queue.Queue()\
            \cf13 self\cf6 ._reducer_callback = \cf5 None\
\
        \cf4 # Internal variables\
        \cf13 self\cf6 ._backend = backend\
        \cf13 self\cf6 ._running = \cf5 False\
        \cf13 self\cf6 ._managed_backend = \cf5 False\
        \cf13 self\cf6 ._id = uuid4().hex\
        \cf13 self\cf6 ._call_ref = \cf5 None\
\
    def \cf12 __enter__\cf6 (\cf13 self\cf6 ):\
        \cf13 self\cf6 ._managed_backend = \cf5 True\
        \cf13 self\cf6 ._calling = \cf5 False\
        \cf13 self\cf6 ._initialize_backend()\
        \cf5 return \cf13 self\
\
    \cf5 def \cf12 __exit__\cf6 (\cf13 self\cf6 , exc_type, exc_value, traceback):\
        \cf13 self\cf6 ._managed_backend = \cf5 False\
        if \cf13 self\cf6 .return_generator \cf5 and \cf13 self\cf6 ._calling:\
            \cf13 self\cf6 ._abort()\
        \cf13 self\cf6 ._terminate_and_reset()\
\
    \cf5 def \cf9 _initialize_backend\cf6 (\cf13 self\cf6 ):\
        
\f0\i \cf2 """Build a process or thread pool and return the number of workers"""\
        
\f1\i0 \cf5 try\cf6 :\
            n_jobs = \cf13 self\cf6 ._backend.configure(\cf10 n_jobs\cf6 =\cf13 self\cf6 .n_jobs, \cf10 parallel\cf6 =\cf13 self\cf6 ,\
                                             **\cf13 self\cf6 ._backend_args)\
            \cf5 if \cf13 self\cf6 .timeout \cf5 is not None and not \cf13 self\cf6 ._backend.supports_timeout:\
                warnings.warn(\
                    \cf8 'The backend class \{!r\} does not support timeout. '\
                    "You have set 'timeout=\{\}' in Parallel but "\
                    "the 'timeout' parameter will not be used."\cf6 .format(\
                        \cf13 self\cf6 ._backend.__class__.\cf12 __name__\cf6 ,\
                        \cf13 self\cf6 .timeout))\
\
        \cf5 except \cf6 FallbackToBackend \cf5 as \cf6 e:\
            \cf4 # Recursively initialize the backend in case of requested fallback.\
            \cf13 self\cf6 ._backend = e.backend\
            n_jobs = \cf13 self\cf6 ._initialize_backend()\
\
        \cf5 return \cf6 n_jobs\
\
    \cf5 def \cf9 _effective_n_jobs\cf6 (\cf13 self\cf6 ):\
        \cf5 if \cf13 self\cf6 ._backend:\
            \cf5 return \cf13 self\cf6 ._backend.effective_n_jobs(\cf13 self\cf6 .n_jobs)\
        \cf5 return \cf11 1\
\
    \cf5 def \cf9 _terminate_and_reset\cf6 (\cf13 self\cf6 ):\
        \cf5 if \cf7 hasattr\cf6 (\cf13 self\cf6 ._backend, \cf8 'stop_call'\cf6 ) \cf5 and \cf13 self\cf6 ._calling:\
            \cf13 self\cf6 ._backend.stop_call()\
        \cf13 self\cf6 ._calling = \cf5 False\
        if not \cf13 self\cf6 ._managed_backend:\
            \cf13 self\cf6 ._backend.terminate()\
\
    \cf5 def \cf9 _dispatch\cf6 (\cf13 self\cf6 , batch):\
        
\f0\i \cf2 """Queue the batch for computing, with or without multiprocessing\
\
        WARNING: this method is not thread-safe: it should be only called\
        indirectly via dispatch_one_batch.\
\
        """\
        
\f1\i0 \cf4 # If job.get() catches an exception, it closes the queue:\
        \cf5 if \cf13 self\cf6 ._aborting:\
            \cf5 return\
\
        \cf6 batch_size = \cf7 len\cf6 (batch)\
\
        \cf13 self\cf6 .n_dispatched_tasks += batch_size\
        \cf13 self\cf6 .n_dispatched_batches += \cf11 1\
\
        \cf6 dispatch_timestamp = time.time()\
\
        batch_tracker = BatchCompletionCallBack(\
            dispatch_timestamp, batch_size, \cf13 self\
        \cf6 )\
        \cf13 self\cf6 ._jobs.append(batch_tracker)\
\
        job = \cf13 self\cf6 ._backend.apply_async(batch, \cf10 callback\cf6 =batch_tracker)\
        batch_tracker.register_job(job)\
\
    \cf5 def \cf9 dispatch_next\cf6 (\cf13 self\cf6 ):\
        
\f0\i \cf2 """Dispatch more data for parallel processing\
\
        This method is meant to be called concurrently by the multiprocessing\
        callback. We rely on the thread-safety of dispatch_one_batch to protect\
        against concurrent consumption of the unprotected iterator.\
\
        """\
        
\f1\i0 \cf5 if not \cf13 self\cf6 .dispatch_one_batch(\cf13 self\cf6 ._original_iterator):\
            \cf13 self\cf6 ._iterating = \cf5 False\
            \cf13 self\cf6 ._original_iterator = \cf5 None\
\
    def \cf9 dispatch_one_batch\cf6 (\cf13 self\cf6 , iterator):\
        
\f0\i \cf2 """Prefetch the tasks for the next batch and dispatch them.\
\
        The effective size of the batch is computed here.\
        If there are no more jobs to dispatch, return False, else return True.\
\
        The iterator consumption and dispatching is protected by the same\
        lock so calling this function should be thread safe.\
\
        """\
\
        
\f1\i0 \cf5 if \cf13 self\cf6 ._aborting:\
            \cf5 return False\
\
        \cf6 batch_size = \cf13 self\cf6 ._get_batch_size()\
\
        \cf5 with \cf13 self\cf6 ._lock:\
            \cf4 # to ensure an even distribution of the workload between workers,\
            # we look ahead in the original iterators more than batch_size\
            # tasks - However, we keep consuming only one batch at each\
            # dispatch_one_batch call. The extra tasks are stored in a local\
            # queue, _ready_batches, that is looked-up prior to re-consuming\
            # tasks from the origal iterator.\
            \cf5 try\cf6 :\
                tasks = \cf13 self\cf6 ._ready_batches.get(\cf10 block\cf6 =\cf5 False\cf6 )\
            \cf5 except \cf6 queue.Empty:\
                \cf4 # slice the iterator n_jobs * batchsize items at a time. If the\
                # slice returns less than that, then the current batchsize puts\
                # too much weight on a subset of workers, while other may end\
                # up starving. So in this case, re-scale the batch size\
                # accordingly to distribute evenly the last items between all\
                # workers.\
                \cf6 n_jobs = \cf13 self\cf6 ._cached_effective_n_jobs\
                big_batch_size = batch_size * n_jobs\
\
                islice = \cf7 list\cf6 (itertools.islice(iterator, big_batch_size))\
                \cf5 if \cf7 len\cf6 (islice) == \cf11 0\cf6 :\
                    \cf5 return False\
                elif \cf6 (iterator \cf5 is \cf13 self\cf6 ._original_iterator \cf5 and\
                      \cf7 len\cf6 (islice) < big_batch_size):\
                    \cf4 # We reached the end of the original iterator (unless\
                    # iterator is the ``pre_dispatch``-long initial slice of\
                    # the original iterator) -- decrease the batch size to\
                    # account for potential variance in the batches running\
                    # time.\
                    \cf6 final_batch_size = \cf7 max\cf6 (\cf11 1\cf6 , \cf7 len\cf6 (islice) // (\cf11 10 \cf6 * n_jobs))\
                \cf5 else\cf6 :\
                    final_batch_size = \cf7 max\cf6 (\cf11 1\cf6 , \cf7 len\cf6 (islice) // n_jobs)\
\
                \cf4 # enqueue n_jobs batches in a local queue\
                \cf5 for \cf6 i \cf5 in \cf7 range\cf6 (\cf11 0\cf6 , \cf7 len\cf6 (islice), final_batch_size):\
                    tasks = BatchedCalls(islice[i:i + final_batch_size],\
                                         \cf13 self\cf6 ._backend.get_nested_backend(),\
                                         \cf13 self\cf6 ._reducer_callback,\
                                         \cf13 self\cf6 ._pickle_cache)\
                    \cf13 self\cf6 ._ready_batches.put(tasks)\
\
                \cf4 # finally, get one task.\
                \cf6 tasks = \cf13 self\cf6 ._ready_batches.get(\cf10 block\cf6 =\cf5 False\cf6 )\
            \cf5 if \cf7 len\cf6 (tasks) == \cf11 0\cf6 :\
                \cf4 # No more tasks available in the iterator: tell caller to stop.\
                \cf5 return False\
            else\cf6 :\
                \cf13 self\cf6 ._dispatch(tasks)\
                \cf5 return True\
\
    def \cf9 _get_batch_size\cf6 (\cf13 self\cf6 ):\
        
\f0\i \cf2 """Returns the effective batch size for dispatch"""\
        
\f1\i0 \cf5 if \cf13 self\cf6 .batch_size == \cf8 'auto'\cf6 :\
            \cf5 return \cf13 self\cf6 ._backend.compute_batch_size()\
        \cf5 else\cf6 :\
            \cf4 # Fixed batch size strategy\
            \cf5 return \cf13 self\cf6 .batch_size\
\
    \cf5 def \cf9 _print\cf6 (\cf13 self\cf6 , msg):\
        
\f0\i \cf2 """Display the message on stout or stderr depending on verbosity"""\
        
\f1\i0 \cf4 # XXX: Not using the logger framework: need to\
        # learn to use logger better.\
        \cf5 if not \cf13 self\cf6 .verbose:\
            \cf5 return\
        if \cf13 self\cf6 .verbose < \cf11 50\cf6 :\
            writer = sys.stderr.write\
        \cf5 else\cf6 :\
            writer = sys.stdout.write\
        writer(\cf8 f"[\cf5 \{\cf13 self\cf5 \}\cf8 ]: \cf5 \{\cf6 msg\cf5 \}\\n\cf8 "\cf6 )\
\
    \cf5 def \cf9 _is_completed\cf6 (\cf13 self\cf6 ):\
        
\f0\i \cf2 """Check if all tasks have been completed"""\
        
\f1\i0 \cf5 return \cf13 self\cf6 .n_completed_tasks == \cf13 self\cf6 .n_dispatched_tasks \cf5 and not \cf6 (\
            \cf13 self\cf6 ._iterating \cf5 or \cf13 self\cf6 ._aborting\
        )\
\
    \cf5 def \cf9 print_progress\cf6 (\cf13 self\cf6 ):\
        
\f0\i \cf2 """Display the process of the parallel execution only a fraction\
           of time, controlled by self.verbose.\
        """\
\
        
\f1\i0 \cf5 if not \cf13 self\cf6 .verbose:\
            \cf5 return\
\
        \cf6 elapsed_time = time.time() - \cf13 self\cf6 ._start_time\
\
        \cf5 if \cf13 self\cf6 ._is_completed():\
            \cf4 # Make sure that we get a last message telling us we are done\
            \cf13 self\cf6 ._print(\
                \cf8 f"Done \cf5 \{\cf13 self\cf6 .n_completed_tasks\cf5 :\cf8 3d\cf5 \}\cf8  out of "\
                f"\cf5 \{\cf13 self\cf6 .n_completed_tasks\cf5 :\cf8 3d\cf5 \}\cf8  | elapsed: "\
                f"\cf5 \{\cf6 short_format_time(elapsed_time)\cf5 \}\cf8  finished"\
            \cf6 )\
            \cf5 return\
\
        \cf4 # Original job iterator becomes None once it has been fully\
        # consumed : at this point we know the total number of jobs and we are\
        # able to display an estimation of the remaining time based on already\
        # completed jobs. Otherwise, we simply display the number of completed\
        # tasks.\
        \cf5 elif \cf13 self\cf6 ._original_iterator \cf5 is not None\cf6 :\
            \cf5 if \cf6 _verbosity_filter(\cf13 self\cf6 .n_dispatched_batches, \cf13 self\cf6 .verbose):\
                \cf5 return\
            \cf13 self\cf6 ._print(\
                \cf8 f"Done \cf5 \{\cf13 self\cf6 .n_completed_tasks\cf5 :\cf8 3d\cf5 \}\cf8  tasks      | elapsed: "\
                f"\cf5 \{\cf6 short_format_time(elapsed_time)\cf5 \}\cf8 "\
            \cf6 )\
        \cf5 else\cf6 :\
            index = \cf13 self\cf6 .n_completed_tasks\
            \cf4 # We are finished dispatching\
            \cf6 total_tasks = \cf13 self\cf6 .n_dispatched_tasks\
            \cf4 # We always display the first loop\
            \cf5 if not \cf6 index == \cf11 0\cf6 :\
                \cf4 # Display depending on the number of remaining items\
                # A message as soon as we finish dispatching, cursor is 0\
                \cf6 cursor = (total_tasks - index + \cf11 1 \cf6 -\
                          \cf13 self\cf6 ._pre_dispatch_amount)\
                frequency = (total_tasks // \cf13 self\cf6 .verbose) + \cf11 1\
                \cf6 is_last_item = (index + \cf11 1 \cf6 == total_tasks)\
                \cf5 if \cf6 (is_last_item \cf5 or \cf6 cursor % frequency):\
                    \cf5 return\
            \cf6 remaining_time = (elapsed_time / index) * \\\
                             (\cf13 self\cf6 .n_dispatched_tasks - index * \cf11 1.0\cf6 )\
            \cf4 # only display status if remaining time is greater or equal to 0\
            \cf13 self\cf6 ._print(\
                \cf8 f"Done \cf5 \{\cf6 index\cf5 :\cf8 3d\cf5 \}\cf8  out of \cf5 \{\cf6 total_tasks\cf5 :\cf8 3d\cf5 \}\cf8  | elapsed: "\
                f"\cf5 \{\cf6 short_format_time(elapsed_time)\cf5 \}\cf8  remaining: "\
                f"\cf5 \{\cf6 short_format_time(remaining_time)\cf5 \}\cf8 "\
            \cf6 )\
\
    \cf5 def \cf9 _abort\cf6 (\cf13 self\cf6 ):\
        \cf4 # Stop dispatching new jobs in the async callback thread\
        \cf13 self\cf6 ._aborting = \cf5 True\
\
        \cf4 # If the backend allows it, cancel or kill remaining running\
        # tasks without waiting for the results as we will raise\
        # the exception we got back to the caller instead of returning\
        # any result.\
        \cf6 backend = \cf13 self\cf6 ._backend\
        \cf5 if \cf6 (\cf5 not \cf13 self\cf6 ._aborted \cf5 and \cf7 hasattr\cf6 (backend, \cf8 'abort_everything'\cf6 )):\
            \cf4 # If the backend is managed externally we need to make sure\
            # to leave it in a working state to allow for future jobs\
            # scheduling.\
            \cf6 ensure_ready = \cf13 self\cf6 ._managed_backend\
            backend.abort_everything(\cf10 ensure_ready\cf6 =ensure_ready)\
        \cf13 self\cf6 ._aborted = \cf5 True\
\
    def \cf9 _start\cf6 (\cf13 self\cf6 , iterator, pre_dispatch):\
        \cf4 # Only set self._iterating to True if at least a batch\
        # was dispatched. In particular this covers the edge\
        # case of Parallel used with an exhausted iterator. If\
        # self._original_iterator is None, then this means either\
        # that pre_dispatch == "all", n_jobs == 1 or that the first batch\
        # was very quick and its callback already dispatched all the\
        # remaining jobs.\
        \cf13 self\cf6 ._iterating = \cf5 False\
        if \cf13 self\cf6 .dispatch_one_batch(iterator):\
            \cf13 self\cf6 ._iterating = \cf13 self\cf6 ._original_iterator \cf5 is not None\
\
        while \cf13 self\cf6 .dispatch_one_batch(iterator):\
            \cf5 pass\
\
        if \cf6 pre_dispatch == \cf8 "all"\cf6 :\
            \cf4 # The iterable was consumed all at once by the above for loop.\
            # No need to wait for async callbacks to trigger to\
            # consumption.\
            \cf13 self\cf6 ._iterating = \cf5 False\
\
    def \cf9 _get_outputs\cf6 (\cf13 self\cf6 , iterator, pre_dispatch):\
        
\f0\i \cf2 """Iterator returning the tasks' output as soon as they are ready."""\
        
\f1\i0 \cf6 dispatch_thread_id = threading.get_ident()\
        detach_generator_exit = \cf5 False\
        try\cf6 :\
            \cf13 self\cf6 ._start(iterator, pre_dispatch)\
            \cf4 # first yield returns None, for internal use only. This ensures\
            # that we enter the try/except block and start dispatching the\
            # tasks.\
            \cf5 yield\
\
            with \cf13 self\cf6 ._backend.retrieval_context():\
                \cf5 yield from \cf13 self\cf6 ._retrieve()\
\
        \cf5 except \cf7 GeneratorExit\cf6 :\
            \cf4 # The generator has been garbage collected before being fully\
            # consumed. This aborts the remaining tasks if possible and warn\
            # the user if necessary.\
            \cf13 self\cf6 ._exception = \cf5 True\
\
            \cf4 # In some interpreters such as PyPy, GeneratorExit can be raised in\
            # a different thread than the one used to start the dispatch of the\
            # parallel tasks. This can lead to hang when a thread attempts to\
            # join itself. As workaround, we detach the execution of the\
            # aborting code to a dedicated thread. We then need to make sure\
            # the rest of the function does not call `_terminate_and_reset`\
            # in finally.\
            \cf5 if \cf6 dispatch_thread_id != threading.get_ident():\
                \cf5 if not \cf6 IS_PYPY:\
                    warnings.warn(\
                        \cf8 "A generator produced by joblib.Parallel has been "\
                        "gc'ed in an unexpected thread. This behavior should "\
                        "not cause major -issues but to make sure, please "\
                        "report this warning and your use case at "\
                        "https://github.com/joblib/joblib/issues so it can "\
                        "be investigated."\
                    \cf6 )\
\
                detach_generator_exit = \cf5 True\
                \cf6 _parallel = \cf13 self\
\
                \cf5 class \cf6 _GeneratorExitThread(threading.Thread):\
                    \cf5 def \cf9 run\cf6 (\cf13 self\cf6 ):\
                        _parallel._abort()\
                        \cf5 if \cf6 _parallel.return_generator:\
                            _parallel._warn_exit_early()\
                        _parallel._terminate_and_reset()\
\
                _GeneratorExitThread(\
                    \cf10 name\cf6 =\cf8 "GeneratorExitThread"\
                \cf6 ).start()\
                \cf5 return\
\
            \cf4 # Otherwise, we are in the thread that started the dispatch: we can\
            # safely abort the execution and warn the user.\
            \cf13 self\cf6 ._abort()\
            \cf5 if \cf13 self\cf6 .return_generator:\
                \cf13 self\cf6 ._warn_exit_early()\
\
            \cf5 raise\
\
        \cf4 # Note: we catch any BaseException instead of just Exception instances\
        # to also include KeyboardInterrupt\
        \cf5 except \cf7 BaseException\cf6 :\
            \cf13 self\cf6 ._exception = \cf5 True\
            \cf13 self\cf6 ._abort()\
            \cf5 raise\
        finally\cf6 :\
            \cf4 # Store the unconsumed tasks and terminate the workers if necessary\
            \cf6 _remaining_outputs = ([] \cf5 if \cf13 self\cf6 ._exception \cf5 else \cf13 self\cf6 ._jobs)\
            \cf13 self\cf6 ._jobs = collections.deque()\
            \cf13 self\cf6 ._running = \cf5 False\
            if not \cf6 detach_generator_exit:\
                \cf13 self\cf6 ._terminate_and_reset()\
\
        \cf5 while \cf7 len\cf6 (_remaining_outputs) > \cf11 0\cf6 :\
            batched_results = _remaining_outputs.popleft()\
            batched_results = batched_results.get_result(\cf13 self\cf6 .timeout)\
            \cf5 for \cf6 result \cf5 in \cf6 batched_results:\
                \cf5 yield \cf6 result\
\
    \cf5 def \cf9 _wait_retrieval\cf6 (\cf13 self\cf6 ):\
        
\f0\i \cf2 """Return True if we need to continue retriving some tasks."""\
\
        
\f1\i0 \cf4 # If the input load is still being iterated over, it means that tasks\
        # are still on the dispatch wait list and their results will need to\
        # be retrieved later on.\
        \cf5 if \cf13 self\cf6 ._iterating:\
            \cf5 return True\
\
        \cf4 # If some of the dispatched tasks are still being processed by the\
        # workers, wait for the compute to finish before starting retrieval\
        \cf5 if \cf13 self\cf6 .n_completed_tasks < \cf13 self\cf6 .n_dispatched_tasks:\
            \cf5 return True\
\
        \cf4 # For backends that does not support retrieving asynchronously the\
        # result to the main process, all results must be carefully retrieved\
        # in the _retrieve loop in the main thread while the backend is alive.\
        # For other backends, the actual retrieval is done asynchronously in\
        # the callback thread, and we can terminate the backend before the\
        # `self._jobs` result list has been emptied. The remaining results\
        # will be collected in the `finally` step of the generator.\
        \cf5 if not \cf13 self\cf6 ._backend.supports_retrieve_callback:\
            \cf5 if \cf7 len\cf6 (\cf13 self\cf6 ._jobs) > \cf11 0\cf6 :\
                \cf5 return True\
\
        return False\
\
    def \cf9 _retrieve\cf6 (\cf13 self\cf6 ):\
        \cf5 while \cf13 self\cf6 ._wait_retrieval():\
\
            \cf4 # If the callback thread of a worker has signaled that its task\
            # triggered an exception, or if the retrieval loop has raised an\
            # exception (e.g. `GeneratorExit`), exit the loop and surface the\
            # worker traceback.\
            \cf5 if \cf13 self\cf6 ._aborting:\
                \cf13 self\cf6 ._raise_error_fast()\
                \cf5 break\
\
            \cf4 # If the next job is not ready for retrieval yet, we just wait for\
            # async callbacks to progress.\
            \cf5 if \cf6 ((\cf7 len\cf6 (\cf13 self\cf6 ._jobs) == \cf11 0\cf6 ) \cf5 or\
                \cf6 (\cf13 self\cf6 ._jobs[\cf11 0\cf6 ].get_status(\
                    \cf10 timeout\cf6 =\cf13 self\cf6 .timeout) == TASK_PENDING)):\
                time.sleep(\cf11 0.01\cf6 )\
                \cf5 continue\
\
            \cf4 # We need to be careful: the job list can be filling up as\
            # we empty it and Python list are not thread-safe by\
            # default hence the use of the lock\
            \cf5 with \cf13 self\cf6 ._lock:\
                batched_results = \cf13 self\cf6 ._jobs.popleft()\
\
            \cf4 # Flatten the batched results to output one output at a time\
            \cf6 batched_results = batched_results.get_result(\cf13 self\cf6 .timeout)\
            \cf5 for \cf6 result \cf5 in \cf6 batched_results:\
                \cf13 self\cf6 ._nb_consumed += \cf11 1\
                \cf5 yield \cf6 result\
\
    \cf5 def \cf9 _raise_error_fast\cf6 (\cf13 self\cf6 ):\
        
\f0\i \cf2 """If we are aborting, raise if a job caused an error."""\
\
        
\f1\i0 \cf4 # Find the first job whose status is TASK_ERROR if it exists.\
        \cf5 with \cf13 self\cf6 ._lock:\
            error_job = \cf7 next\cf6 ((job \cf5 for \cf6 job \cf5 in \cf13 self\cf6 ._jobs\
                              \cf5 if \cf6 job.status == TASK_ERROR), \cf5 None\cf6 )\
\
        \cf4 # If this error job exists, immediatly raise the error by\
        # calling get_result. This job might not exists if abort has been\
        # called directly or if the generator is gc'ed.\
        \cf5 if \cf6 error_job \cf5 is not None\cf6 :\
            error_job.get_result(\cf13 self\cf6 .timeout)\
\
    \cf5 def \cf9 _warn_exit_early\cf6 (\cf13 self\cf6 ):\
        
\f0\i \cf2 """Warn the user if the generator is gc'ed before being consumned."""\
        
\f1\i0 \cf6 ready_outputs = \cf13 self\cf6 .n_completed_tasks - \cf13 self\cf6 ._nb_consumed\
        is_completed = \cf13 self\cf6 ._is_completed()\
        msg = \cf8 ""\
        \cf5 if \cf6 ready_outputs:\
            msg += (\
                \cf8 f"\cf5 \{\cf6 ready_outputs\cf5 \}\cf8  tasks have been successfully executed "\
                " but not used."\
            \cf6 )\
            \cf5 if not \cf6 is_completed:\
                msg += \cf8 " Additionally, "\
\
        \cf5 if not \cf6 is_completed:\
            msg += (\
                \cf8 f"\cf5 \{\cf13 self\cf6 .n_dispatched_tasks - \cf13 self\cf6 .n_completed_tasks\cf5 \}\cf8  tasks "\
                "which were still being processed by the workers have been "\
                "cancelled."\
            \cf6 )\
\
        \cf5 if \cf6 msg:\
            msg += (\
                \cf8 " You could benefit from adjusting the input task "\
                "iterator to limit unnecessary computation time."\
            \cf6 )\
\
            warnings.warn(msg)\
\
    \cf5 def \cf9 _get_sequential_output\cf6 (\cf13 self\cf6 , iterable):\
        
\f0\i \cf2 """Separate loop for sequential output.\
\
        This simplifies the traceback in case of errors and reduces the\
        overhead of calling sequential tasks with `joblib`.\
        """\
        
\f1\i0 \cf5 try\cf6 :\
            \cf13 self\cf6 ._iterating = \cf5 True\
            \cf13 self\cf6 ._original_iterator = iterable\
            batch_size = \cf13 self\cf6 ._get_batch_size()\
\
            \cf5 if \cf6 batch_size != \cf11 1\cf6 :\
                it = \cf7 iter\cf6 (iterable)\
                iterable_batched = \cf7 iter\cf6 (\
                    \cf5 lambda\cf6 : \cf7 tuple\cf6 (itertools.islice(it, batch_size)), ()\
                )\
                iterable = (\
                    task \cf5 for \cf6 batch \cf5 in \cf6 iterable_batched \cf5 for \cf6 task \cf5 in \cf6 batch\
                )\
\
            \cf4 # first yield returns None, for internal use only. This ensures\
            # that we enter the try/except block and setup the generator.\
            \cf5 yield None\
\
            \cf4 # Sequentially call the tasks and yield the results.\
            \cf5 for \cf6 func, args, kwargs \cf5 in \cf6 iterable:\
                \cf13 self\cf6 .n_dispatched_batches += \cf11 1\
                \cf13 self\cf6 .n_dispatched_tasks += \cf11 1\
                \cf6 res = func(*args, **kwargs)\
                \cf13 self\cf6 .n_completed_tasks += \cf11 1\
                \cf13 self\cf6 .print_progress()\
                \cf5 yield \cf6 res\
                \cf13 self\cf6 ._nb_consumed += \cf11 1\
        \cf5 except \cf7 BaseException\cf6 :\
            \cf13 self\cf6 ._exception = \cf5 True\
            \cf13 self\cf6 ._aborting = \cf5 True\
            \cf13 self\cf6 ._aborted = \cf5 True\
            raise\
        finally\cf6 :\
            \cf13 self\cf6 .print_progress()\
            \cf13 self\cf6 ._running = \cf5 False\
            \cf13 self\cf6 ._iterating = \cf5 False\
            \cf13 self\cf6 ._original_iterator = \cf5 None\
\
    def \cf9 _reset_run_tracking\cf6 (\cf13 self\cf6 ):\
        
\f0\i \cf2 """Reset the counters and flags used to track the execution."""\
\
        
\f1\i0 \cf4 # Makes sur the parallel instance was not previously running in a\
        # thread-safe way.\
        \cf5 with \cf7 getattr\cf6 (\cf13 self\cf6 , \cf8 '_lock'\cf6 , nullcontext()):\
            \cf5 if \cf13 self\cf6 ._running:\
                msg = \cf8 'This Parallel instance is already running !'\
                \cf5 if \cf13 self\cf6 .return_generator \cf5 is True\cf6 :\
                    msg += (\
                        \cf8 " Before submitting new tasks, you must wait for the "\
                        "completion of all the previous tasks, or clean all "\
                        "references to the output generator."\
                    \cf6 )\
                \cf5 raise \cf7 RuntimeError\cf6 (msg)\
            \cf13 self\cf6 ._running = \cf5 True\
\
        \cf4 # Counter to keep track of the task dispatched and completed.\
        \cf13 self\cf6 .n_dispatched_batches = \cf11 0\
        \cf13 self\cf6 .n_dispatched_tasks = \cf11 0\
        \cf13 self\cf6 .n_completed_tasks = \cf11 0\
\
        \cf4 # Following count is incremented by one each time the user iterates\
        # on the output generator, it is used to prepare an informative\
        # warning message in case the generator is deleted before all the\
        # dispatched tasks have been consumed.\
        \cf13 self\cf6 ._nb_consumed = \cf11 0\
\
        \cf4 # Following flags are used to synchronize the threads in case one of\
        # the tasks error-out to ensure that all workers abort fast and that\
        # the backend terminates properly.\
\
        # Set to True as soon as a worker signals that a task errors-out\
        \cf13 self\cf6 ._exception = \cf5 False\
        \cf4 # Set to True in case of early termination following an incident\
        \cf13 self\cf6 ._aborting = \cf5 False\
        \cf4 # Set to True after abortion is complete\
        \cf13 self\cf6 ._aborted = \cf5 False\
\
    def \cf12 __call__\cf6 (\cf13 self\cf6 , iterable):\
        
\f0\i \cf2 """Main function to dispatch parallel tasks."""\
\
        
\f1\i0 \cf13 self\cf6 ._reset_run_tracking()\
        \cf13 self\cf6 ._start_time = time.time()\
\
        \cf5 if not \cf13 self\cf6 ._managed_backend:\
            n_jobs = \cf13 self\cf6 ._initialize_backend()\
        \cf5 else\cf6 :\
            n_jobs = \cf13 self\cf6 ._effective_n_jobs()\
\
        \cf5 if \cf6 n_jobs == \cf11 1\cf6 :\
            \cf4 # If n_jobs==1, run the computation sequentially and return\
            # immediatly to avoid overheads.\
            \cf6 output = \cf13 self\cf6 ._get_sequential_output(iterable)\
            \cf7 next\cf6 (output)\
            \cf5 return \cf6 output \cf5 if \cf13 self\cf6 .return_generator \cf5 else \cf7 list\cf6 (output)\
\
        \cf4 # Let's create an ID that uniquely identifies the current call. If the\
        # call is interrupted early and that the same instance is immediately\
        # re-used, this id will be used to prevent workers that were\
        # concurrently finalizing a task from the previous call to run the\
        # callback.\
        \cf5 with \cf13 self\cf6 ._lock:\
            \cf13 self\cf6 ._call_id = uuid4().hex\
\
        \cf4 # self._effective_n_jobs should be called in the Parallel.__call__\
        # thread only -- store its value in an attribute for further queries.\
        \cf13 self\cf6 ._cached_effective_n_jobs = n_jobs\
\
        \cf5 if \cf7 isinstance\cf6 (\cf13 self\cf6 ._backend, LokyBackend):\
            \cf4 # For the loky backend, we add a callback executed when reducing\
            # BatchCalls, that makes the loky executor use a temporary folder\
            # specific to this Parallel object when pickling temporary memmaps.\
            # This callback is necessary to ensure that several Parallel\
            # objects using the same resuable executor don't use the same\
            # temporary resources.\
\
            \cf5 def \cf9 _batched_calls_reducer_callback\cf6 ():\
                \cf4 # Relevant implementation detail: the following lines, called\
                # when reducing BatchedCalls, are called in a thread-safe\
                # situation, meaning that the context of the temporary folder\
                # manager will not be changed in between the callback execution\
                # and the end of the BatchedCalls pickling. The reason is that\
                # pickling (the only place where set_current_context is used)\
                # is done from a single thread (the queue_feeder_thread).\
                \cf13 self\cf6 ._backend._workers._temp_folder_manager.set_current_context(  \cf4 # noqa\
                    \cf13 self\cf6 ._id\
                )\
            \cf13 self\cf6 ._reducer_callback = _batched_calls_reducer_callback\
\
        \cf4 # self._effective_n_jobs should be called in the Parallel.__call__\
        # thread only -- store its value in an attribute for further queries.\
        \cf13 self\cf6 ._cached_effective_n_jobs = n_jobs\
\
        backend_name = \cf13 self\cf6 ._backend.__class__.\cf12 __name__\
        \cf5 if \cf6 n_jobs == \cf11 0\cf6 :\
            \cf5 raise \cf7 RuntimeError\cf6 (\cf8 "%s has no active worker." \cf6 % backend_name)\
\
        \cf13 self\cf6 ._print(\
            \cf8 f"Using backend \cf5 \{\cf6 backend_name\cf5 \}\cf8  with \cf5 \{\cf6 n_jobs\cf5 \}\cf8  concurrent workers."\
        \cf6 )\
        \cf5 if \cf7 hasattr\cf6 (\cf13 self\cf6 ._backend, \cf8 'start_call'\cf6 ):\
            \cf13 self\cf6 ._backend.start_call()\
\
        \cf4 # Following flag prevents double calls to `backend.stop_call`.\
        \cf13 self\cf6 ._calling = \cf5 True\
\
        \cf6 iterator = \cf7 iter\cf6 (iterable)\
        pre_dispatch = \cf13 self\cf6 .pre_dispatch\
\
        \cf5 if \cf6 pre_dispatch == \cf8 'all'\cf6 :\
            \cf4 # prevent further dispatch via multiprocessing callback thread\
            \cf13 self\cf6 ._original_iterator = \cf5 None\
            \cf13 self\cf6 ._pre_dispatch_amount = \cf11 0\
        \cf5 else\cf6 :\
            \cf13 self\cf6 ._original_iterator = iterator\
            \cf5 if \cf7 hasattr\cf6 (pre_dispatch, \cf8 'endswith'\cf6 ):\
                pre_dispatch = eval_expr(\
                    pre_dispatch.replace(\cf8 "n_jobs"\cf6 , \cf7 str\cf6 (n_jobs))\
                )\
            \cf13 self\cf6 ._pre_dispatch_amount = pre_dispatch = \cf7 int\cf6 (pre_dispatch)\
\
            \cf4 # The main thread will consume the first pre_dispatch items and\
            # the remaining items will later be lazily dispatched by async\
            # callbacks upon task completions.\
\
            # TODO: this iterator should be batch_size * n_jobs\
            \cf6 iterator = itertools.islice(iterator, \cf13 self\cf6 ._pre_dispatch_amount)\
\
        \cf4 # Use a caching dict for callables that are pickled with cloudpickle to\
        # improve performances. This cache is used only in the case of\
        # functions that are defined in the __main__ module, functions that\
        # are defined locally (inside another function) and lambda expressions.\
        \cf13 self\cf6 ._pickle_cache = \cf7 dict\cf6 ()\
\
        output = \cf13 self\cf6 ._get_outputs(iterator, pre_dispatch)\
        \cf13 self\cf6 ._call_ref = weakref.ref(output)\
\
        \cf4 # The first item from the output is blank, but it makes the interpreter\
        # progress until it enters the Try/Except block of the generator and\
        # reach the first `yield` statement. This starts the aynchronous\
        # dispatch of the tasks to the workers.\
        \cf7 next\cf6 (output)\
\
        \cf5 return \cf6 output \cf5 if \cf13 self\cf6 .return_generator \cf5 else \cf7 list\cf6 (output)\
\
    \cf5 def \cf12 __repr__\cf6 (\cf13 self\cf6 ):\
        \cf5 return \cf8 '%s(n_jobs=%s)' \cf6 % (\cf13 self\cf6 .__class__.\cf12 __name__\cf6 , \cf13 self\cf6 .n_jobs)\
\
}