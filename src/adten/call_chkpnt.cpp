// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
#include<ad_tensor/chkpnt.hpp>
#include<ad_tensor/adten.hpp>
/*
{xrst_begin call_chkpnt usr}

Calling An Atomic Function
##########################
{xrst_spell
    adomain
    arange
}

Prototype
*********
{xrst_literal ,
    // BEGIN_CALL_CHKPNT, END_CALL_CHKPNT
}

chkpnt_id
*********
is the atomic function identifier; see :ref:`chkpnt_global@store@chkpnt_id` .

adomain
*******
is the AD tensor version of the domain for this atomic function call.

arange
******
is the AD tensor version of the range for this atomic function call.

{xrst_end call_chkpnt}
*/

namespace ad_tensor {
    // BEGIN_CALL_CHKPNT
    vector<adten_t> adten_t::call_chkpnt(
        size_t                 chkpnt_id ,
        const vector<adten_t>& adomain   )
    {   // END_CALL_CHKPNT
        //
        // atom_id
        chkpnt_global_t&  global  = chkpnt_global_t::singleton();
        size_t            atom_id = global.get_atom_id();
        //
        size_t call_info = chkpnt_id;
        return call_atom(atom_id, call_info, adomain);
    }
}
