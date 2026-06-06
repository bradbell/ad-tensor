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
    int
    len
}

Converting size_t Pointer to an ArrayRef
########################################

Prototype
*********
{xrst_literal ,
    include/ad_tensor/dev/size_ptr2array_ref.hpp
    BEGIN_SIZE_PTR2ARRAY_REF, END_SIZE_PTR2ARRAY_REF
}

lock
****
Calls to size_ptr2array_ref must be made in pairs.
The first (second) call should have lock true (lock false).

size_ptr
********
The number of elements in the array is len = ``size_ptr[0]``
The elements in the array are ``size_ptr[1]`` , ... , ``size_ptr[len]`` .
This is not used and should not be in call when lock is false.

array_ref
*********
This is an array ref for a vector<int64_t> that contains the array values.
This is only valid for the current thread, and only until the next
call to size_ptr2array_ref.
In addition, it only valid until the following call to size_ptr2array_ref
with lock false.

{xrst_end size_ptr2array_ref}
*/
//
namespace ad_tensor { namespace dev {
    c10::IntArrayRef size_ptr2array_ref(
        bool          lock     ,
        const size_t* size_ptr ) {
    //
    // locked
    thread_local bool locked = false;
    //
    // array
    thread_local vector<int64_t> array;
    //
    // locked
    if( ! lock )
    {   assert( locked && "size_ptr2array_ref: "
            "a call with lock false was not preceded by a call with lock true"
        );
        locked = false;
        return c10::IntArrayRef();
    }
    assert( ! locked && "size_ptr2array_ref: "
        "attempt to get a lock while another call is holding its lock"
    );
    locked = true;
    //
#ifdef NDEBUG
    array.resize(0)
#else
    size_t cap = array.capacity();
    array.resize(0);
    assert( array.capacity() == cap && "int_array_from_op: "
            "std::vector::resize to zero changed capacity"
    );
# endif
    //
    // size
    size_t size = size_ptr[0];
    assert( size < 20 && "size_t_ptr2array_ref: size >- 20");
    //
    //
    array.resize(size);
    for(size_t i = 0; i < size; ++i) {
        array[i] = static_cast<int64_t>( size_ptr[i + 1] );
    }
    return c10::IntArrayRef (array);
} } }
