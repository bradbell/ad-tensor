// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
#include<ad_tensor/chkpnt.hpp>
#include<ad_tensor/adten.hpp>

namespace ad_tensor {
    vector<adten_t> adten_t::call_chkpnt(
        size_t                 chkpnt_id ,
        const vector<adten_t>& adomain
    ) {
        //
        // atom_id
        chkpnt_global_t&  global  = chkpnt_global_t::singleton();
        size_t            atom_id = global.get_atom_id();
        //
        size_t call_info = chkpnt_id;
        return call_atom(atom_id, call_info, adomain);
    }
}
