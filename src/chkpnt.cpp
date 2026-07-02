// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
/*
{xrst_begin_parent chkpnt usr}

Checkpoint Functions
####################
If a function is used many times, making it a checkpoint
may greatly reduce the amount of memory required to compute derivatives.
This is because only one copy of the functions
constants, parameters and variables are needed.
On the other hand, a checkpoint function must recompute its dependent
parameter and variable values for each derivative direction.

{xrst_end chkpnt}
-----------------------------------------------------------------------------
{xrst_begin make_chkpnt usr}
{xrst_spell
    adfn
}

Convert an AD Function to a Checkpoint Function
###############################################

Syntax
******
{xrst_code cpp}
chkpnt_id = make_chkpnt(adfn)
{xrst_code}

Prototype
*********
{xrst_literal ,
    BEGIN_MAKE_CHKPNT, END_MAKE_CHKPNT
}

adfn
****
On input, this as an AD function representation of the checkpoint function.
Upon return, this AD function is empty.
The representation was moved to global data
and can be accessed using chkpnt_id.

chkpnt_id
*********
is the identifier for this checkpoint function.


{xrst_end make_chkpnt}
-----------------------------------------------------------------------------
{xrst_begin call_chkpnt usr}

Calling A Checkpoint Function
#############################
{xrst_spell
    adomain
    arange
}

Syntax
******
{xrst_code cpp}
arange = call_chkpnt(chkpnt_id, adomain)
{xrst_code}

Prototype
*********
{xrst_literal ,
    // BEGIN_CALL_CHKPNT, END_CALL_CHKPNT
}

chkpnt_id
*********
is the checkpoint function identifier; see :ref:`make_chkpnt@chkpnt_id` .

adomain
*******
is the AD tensor version of the domain for this checkpoint function call.

arange
******
is the AD tensor version of the range for this checkpoint function call.

{xrst_end call_chkpnt}
*/
#include <ad_tensor/dev/chkpnt.hpp>
#include <ad_tensor/dev/move_swap.hpp>
//
//
namespace ad_tensor { // BEGIN_AD_TENSOR_NAMESPACE
//
// BEGIN_MAKE_CHKPNT
size_t make_chkpnt(adfn_t& adfn)
{   // END_MAKE_CHKPNT
    //
    // chkpnt_id
    dev::chkpnt_global_t& chkpnt_global = dev::chkpnt_global_t::singleton();
    dev::chkpnt_info_t    chkpnt_info   = dev::chkpnt_info_t::from_adfn(adfn);
    size_t chkpnt_id                    = chkpnt_global.store(chkpnt_info);
    //
    return chkpnt_id;
}
//
// BEGIN_CALL_CHKPNT
vector<adten_t> call_chkpnt(
    size_t                 chkpnt_id ,
    const vector<adten_t>& adomain   )
{   // END_CALL_CHKPNT
    //
    // atom_id
    dev::chkpnt_global_t&  global  = dev::chkpnt_global_t::singleton();
    size_t                 atom_id = global.get_atom_id();
    //
    size_t call_info = chkpnt_id;
    return call_atom(atom_id, call_info, adomain);
}

}  // END_AD_TENSOR_NAMESPACE
