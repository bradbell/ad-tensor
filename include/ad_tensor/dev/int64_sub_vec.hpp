#pragma once
// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
/*
{xrst_begin int64_sub_vec dev}
{xrst_spell
    dev
}

Sub-vectors of vector<int64_t>
##############################

Syntax
******
{xrst_code cpp}
    int_array_ref = ad_tensor::dev::int64_sub_vec(offset, int64_vec)
{xrst_code}

Prototype
*********
{xrst_literal ,
    BEGIN_INT64_SUB_VEC, END_INT64_SUB_VEC
}

offset
******
is the index in int64_vec where the size of the sub-vector is located.
The elements of the sub-vector start at offset + 1.

int64_vec
*********
the vector that contains the size followed by the elements of the sub-vector.

int_array_ref
*************
is an array reference to the sub-vector.
You can convert this to a vector<int64_t> using as follows:
{xrst_code cpp}
    int_array_ref.vec()
{xrst_code}

{xrst_end int64_sub_vec}
*/
#include <ad_tensor/vector.hpp>
#include <torch/torch.h>
//
namespace ad_tensor { namespace dev {
    // BEGIN_INT64_SUB_VEC
    static c10::IntArrayRef int64_sub_vec(
        size_t offset, const vector<int64_t>& int64_vec
    )
    {   // END_INT64_SUB_VEC
        size_t   size   = size_t( int64_vec[offset] );
        const int64_t* begin  = int64_vec.data() + offset + 1;
        return c10::IntArrayRef(begin, size);
    }

} }
