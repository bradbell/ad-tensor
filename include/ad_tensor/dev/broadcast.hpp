#pragma once
// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
#include <torch/torch.h>
//
// BEGIN_BROADCAST
namespace ad_tensor { namespace dev {
    // dim = broadcast(lock, res, arg)
    c10::IntArrayRef broadcast(
        bool                       lock                               ,
        const c10::IntArrayRef&    res_shape  = c10::IntArrayRef() ,
        const c10::IntArrayRef&    arg_shape  = c10::IntArrayRef()
    );
} }
// END_BROADCAST
