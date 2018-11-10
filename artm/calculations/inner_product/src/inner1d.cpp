#include <inner1d.hpp>

py::array_t<double> memory_efficient_inner1d(
    py::array_t<double> fst_arr, py::array_t<int> fst_indices,
    py::array_t<double> snd_arr, py::array_t<int> snd_indices
) {
    auto fst_arr_ = fst_arr.unchecked<2>();
    auto fst_indices_ = fst_indices.unchecked<1>();
    auto snd_arr_ = snd_arr.unchecked<2>();
    auto snd_indices_ = snd_indices.unchecked<1>();

    auto result = py::array_t<double>(fst_indices_.shape(0));
    auto result_ = result.mutable_unchecked<1>();
    for (ssize_t i = 0; i < fst_indices_.shape(0); ++i) {
        auto fst_index = fst_indices_(i);
        auto snd_index = snd_indices_(i);
        double sum = 0;
        for (ssize_t j = 0; j < fst_arr_.shape(1); ++j) {
            sum += fst_arr_(fst_index, j) * snd_arr_(snd_index, j);
        }
        result_(i) = sum;
    }

    return result;
}
