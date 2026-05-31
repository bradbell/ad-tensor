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
{xrst_spell
    len
}

Converting size_t Pointer to an ArrayRef
########################################

Prototype
*********
{xrst_literal ,
    BEGIN_SIZE_PTR2ARRAY_REF, END_SIZE_PTR2ARRAY_REF
}

size_ptr
********
The number of elements in the array is len = ``size_ptr[0]``
The elements in the array are ``size_ptr[1]`` , ... , ``size_ptr[len]`` .

array_ref
*********
This is an array ref for a vector<long> that contains the array values.
This is only valid for the current thread, and only until the next
call to size_ptr2array_ref.

{xrst_end size_ptr2array_ref}
*/
//
// BEGIN_SIZE_PTR2ARRAY_REF
namespace ad_tensor { namespace dev {
    c10::ArrayRef<long> size_ptr2array_ref(const size_t* size_ptr )
//END_SIZE_PTR2ARRAY_REF
{   //
    // dim
    thread_local vector<long> dim;
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
        dim[i] = static_cast<long>( size_ptr[i + 1] );
    }
    return c10::ArrayRef<long> (dim);
} } }
