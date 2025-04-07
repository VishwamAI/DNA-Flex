#include "fasta_iterator_pybind.h"
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace dnaflex {
namespace parsers {

void init_fasta_iterator(py::module_& m) {
    py::class_<FastaEntry>(m, "FastaEntry")
        .def(py::init<>())
        .def_readwrite("header", &FastaEntry::header)
        .def_readwrite("sequence", &FastaEntry::sequence);

    py::class_<FastaIterator>(m, "FastaIterator")
        .def(py::init<const std::string&>())
        .def("next", &FastaIterator::next)
        .def("reset", &FastaIterator::reset)
        .def("is_valid", &FastaIterator::is_valid)
        .def("__iter__", [](FastaIterator& self) { return &self; })
        .def("__next__", [](FastaIterator& self) {
            FastaEntry entry;
            if (self.next(entry)) {
                return entry;
            }
            throw py::stop_iteration();
        });
}

} // namespace parsers
} // namespace dnaflex