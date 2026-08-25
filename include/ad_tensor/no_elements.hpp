#pragma once
// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
/*
{xrst_begin no_elements usr}

No Element Tensors
##################

Create a no element tensor
**************************
{xrst_literal ,
    BEGIN_CREATE, END_CREATE
}

Check for a no element tensor
*****************************
{xrst_literal ,
    BEGIN_CHECK_NO, END_CHECK_NO
}

Check for a tensor has elements
*******************************
{xrst_literal ,
    BEGIN_CHECK_HAS, END_CHECK_HAS
}


{xrst_end no_elements}
*/
namespace ad_tensor {
    //
    // BEGIN_CREATE
    inline at::Tensor no_elements(void)
    { return torch::empty( {0} ); };
    // END_CREATE
    //
    // BEGIN_CHECK_NO
    template <class TensorType>
    bool no_elements(const TensorType& ten)
    {   return ten.numel() == 0; }
    // END_CHECK_NO
    //
    // BEGIN_CHECK_HAS
    template <class TensorType>
    bool has_elements(const TensorType& ten)
    {   return ten.numel() != 0; }
    // END_CHECK_HAS
}
