#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "msa_profile_pybind.h"
#include "msa_conversion_pybind.h"
#include "cif_dict_pybind.h"
#include "fasta_iterator_pybind.h"

PYBIND11_MODULE(parsers_cpp, m)
{
    m.doc() = R"pbdoc(
        DNA-Flex C++ accelerated parsers module
        -------------------------------------
        
        This module provides C++ implementations of performance-critical parsing
        operations for DNA sequences and structures.
    )pbdoc";
    
    dnaflex::parsers::init_msa_profile(m);
    dnaflex::parsers::init_msa_conversion(m);
    dnaflex::parsers::init_cif_dict(m);
    dnaflex::parsers::init_fasta_iterator(m);
}