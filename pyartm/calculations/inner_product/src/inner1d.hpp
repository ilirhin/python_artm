#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

py::array_t<double> memory_efficient_inner1d(
    py::array_t<double> fst_arr, py::array_t<int> fst_indices,
    py::array_t<double> snd_arr, py::array_t<int> snd_indices
);

PYBIND11_MODULE(cpp_impl, module) {
    module.def("memory_efficient_inner1d", &memory_efficient_inner1d);
}
