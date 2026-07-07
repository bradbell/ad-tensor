#pragma once
// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
#include<ad_tensor/adfn.hpp>
#include<ad_tensor/adten.hpp>
#include<ad_tensor/direction.hpp>
//
namespace ad_tensor {
    //
    // BEGIN_MAKE_CHKPNT
    size_t make_chkpnt(
        adfn_t&                      adfn,
        const vector<at::Tensor>&    domain     = vector<at::Tensor>(),
        c10::ArrayRef<direction_t>   directions = c10::ArrayRef<direction_t>()
    );
    // END_MAKE_CHKPNT
    //
    // call_chkpnt
    vector<adten_t> call_chkpnt(
        size_t                 chkpnt_id ,
        const vector<adten_t>& adomain
    );
}
