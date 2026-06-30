// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
/*
{xrst_begin_parent chkpnt usr}

Checkpoint Functions
####################
If a function is used many times, making it an checkpoint
may greatly reduce memory usage.
because only on copy of its parameters and variables will be required.
On the other hand, its dependent parameter and variable values will
be recomputed for each derivative direction.

{xrst_toc_table after
    src/adten/call_chkpnt.cpp
}

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
    include/ad_tensor/chkpnt.hpp
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
*/
#include <ad_tensor/dev/chkpnt.hpp>
#include <ad_tensor/dev/move_swap.hpp>
//
//
namespace ad_tensor { // BEGIN_AD_TENSOR_NAMESPACE
//
// make_chkpnt
size_t make_chkpnt(adfn_t& adfn) {
    //
    // chkpnt_id
    dev::chkpnt_global_t& chkpnt_global = dev::chkpnt_global_t::singleton();
    dev::chkpnt_info_t    chkpnt_info   = dev::chkpnt_info_t::from_adfn(adfn);
    size_t chkpnt_id                    = chkpnt_global.store(chkpnt_info);
    //
    return chkpnt_id;
}
}  // END_AD_TENSOR_NAMESPACE
