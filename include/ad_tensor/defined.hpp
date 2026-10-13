#pragma once
// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
/*
{xrst_begin defined usr}

Defined and Undefined Tensors
#############################

Create an Undefined Tensor
**************************
{xrst_literal ,
    BEGIN_UNDEFINED, END_UNDEFINED
}

Check for an Undefined tensor
*****************************
{xrst_literal ,
    BEGIN_CHECK_UNDEFINED, END_CHECK_UNDEFINED
}

Check for a Defined tensor
**************************
{xrst_literal ,
    BEGIN_CHECK_DEFINED, END_CHECK_DEFINED
}


{xrst_end defined}
*/
namespace ad_tensor {
    //
    // BEGIN_UNDEFINED
    inline at::Tensor not_defined(void)
    { return at::Tensor(); }
    // END_UNDEFINED
    //
    // BEGIN_CHECK_UNDEFINED
    template <class TensorType>
    bool not_defined(const TensorType& ten)
    {   return ! ten.defined(); }
    // END_CHECK_UNDEFINED
    //
    // BEGIN_CHECK_DEFINED
    template <class TensorType>
    bool is_defined(const TensorType& ten)
    {   return ten.defined(); }
    // END_CHECK_DEFINED
}
