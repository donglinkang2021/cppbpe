PYINC=$(python3 -m pybind11 --includes)
EXTSUF=$(python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
g++ -O3 -Wall -shared -std=c++17 -fPIC $PYINC bpe.cpp -o bpe_cpp${EXTSUF}
