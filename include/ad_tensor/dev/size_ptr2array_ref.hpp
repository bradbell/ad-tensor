#pragma once
// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
#include <torch/torch.h>
//
// BEGIN_SIZE_PTR2ARRAY_REF
namespace ad_tensor { namespace dev {
    // array_ref = size_ptr2array_ref(lock, size_ptr)
    c10::IntArrayRef size_ptr2array_ref(
        bool                     lock    ,
        const size_t* size_ptr = nullptr
    );
} }
// END_SIZE_PTR2ARRAY_REF
