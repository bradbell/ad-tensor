// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
#include <torch/torch.h>
#include <cassert>
#include <ad_tensor/vector.hpp>
#include <ad_tensor/dev/size_ptr2array_ref.hpp>
/*
{xrst_begin size_ptr2array_ref dev}

Converting size_t Pointer to an ArrayRef
########################################

Prototype
*********
{xrst_literal ,
    BEGIN_SIZE_PTR2ARRAY_REF, END_SIZE_PTR2ARRAY_REF
}



{xrst_end size_ptr2array_ref}
*/
//
// BEGIN_SIZE_PTR2ARRAY_REF
namespace ad_tensor { namespace dev {
    c10::ArrayRef<long int> size_ptr2array_ref(const size_t* size_ptr )
//END_SIZE_PTR2ARRAY_REF
{   //
    // dim
    thread_local vector<long int> dim;
#ifdef NDEBUG
    dim.resize(0)
#else
    size_t cap = dim.capacity();
    dim.resize(0);
    assert( dim.capacity() == cap && "int_array_from_op: "
            "std::vector::resize to zero changed capacity"
    );
# endif
    //
    // n_dim
    size_t n_dim = size_ptr[0];
    //
    // dim
    assert( n_dim < 20 && "size_t_ptr2array_ref: n_dim >- 20");
    dim.resize(n_dim);
    for(size_t i = 0; i < n_dim; ++i) {
        dim[i] = static_cast<long int>( size_ptr[i] );
    }
    return c10::ArrayRef<long int> (dim);
} } }
