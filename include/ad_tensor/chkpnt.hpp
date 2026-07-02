#pragma once
// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
#include<ad_tensor/adfn.hpp>
#include<ad_tensor/adten.hpp>
namespace ad_tensor {
    // BEGIN_MAKE_CHKPNT
    size_t make_chkpnt(adfn_t& adfn);
    // END_MAKE_CHKPNT
    vector<adten_t> call_chkpnt(
        size_t                 chkpnt_id ,
        const vector<adten_t>& adomain
    );
}
