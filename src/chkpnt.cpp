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
parameters and variables values for each derivative direction.

{xrst_end chkpnt}
-----------------------------------------------------------------------------
{xrst_begin chkpnt_examples usr}

Checkpoint Examples
###################
{xrst_toc_table
    examples/chkpnt/get_started.cpp
    examples/chkpnt/ad_forward_der.cpp
    examples/chkpnt/ad_reverse_der.cpp
}

{xrst_end chkpnt_examples}
-----------------------------------------------------------------------------
{xrst_begin make_chkpnt usr}
{xrst_spell
    adfn
    dir
    der
    adten
}

Convert an AD Function to a Checkpoint Function
###############################################

direction_t
***********
{xrst_literal ,
    include/ad_tensor/direction.hpp
    BEGIN_DIRECTION, END_DIRECTION
}

Syntax
******
{xrst_code cpp}
chkpnt_id = adten_t::make_chkpnt(adfn, domain, directions)
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
The representation is moved to global data
and can be accessed using chkpnt_id.

chkpnt_id
*********
is the identifier for this checkpoint function.

domain
******
This is only used when directions.size() > 0.
It is a vector of domain values used to record the forward the computation
of the forward or reverse mode derivative of adfn.

directions
**********
Let n_dir = directions.size().
If g is an adfn recorded using calls to this chkpnt function,
this checkpoint will support all the g operations except possibly that
vector<adten_t> versions of g.forward_der and g.reverse_der.
The this checkpoint will support recording up to order n_dir
derivatives of g and evaluation of order n_dir+1 derivatives.

forward
=======
If directions[0] == forward,
this checkpoint call will support recording up to order n_dir
derivatives of g that start a g.forward_der.
This is accomplished by recording adfn_new where
adfn_new.forward_var corresponds to adfn.forward_der.

#.  The new AD function is used to define a new checkpoint that
    supports directions.slice(1) derivatives.

#.  The name of the new AD function is adfn.get_name() + "_forward" .

#.  The trace setting for the new AD function is adfn.get_trace()

reverse
=======
If directions[0] == reverse,
this checkpoint call will support recording order n_dir
derivatives of g that start a g.reverse_der (for vector<adten_t>).
This is accomplished in a similar way to the description above with
forward replaced by reverse.

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
#include <ad_tensor/dev/move_swap.hpp>
#include <ad_tensor/dev/chkpnt.hpp>
#include <ad_tensor/chkpnt.hpp>
//
//
namespace ad_tensor { // BEGIN_AD_TENSOR_NAMESPACE
//
// make_chkpnt
size_t make_chkpnt(
    adfn_t&                          adfn,
    const vector<at::Tensor>&        domain,
    const c10::ArrayRef<direction_t> directions ) {
    return adfn_t::make_chkpnt(adfn, domain, directions );
}
size_t adfn_t::make_chkpnt(
    adfn_t&                          adfn,
    const vector<at::Tensor>&        domain,
    const c10::ArrayRef<direction_t> directions ) {
    //
    // info
    dev::chkpnt_info_t info;
    //
    // info.m_for_chkpnt_id
    if( directions.size() > 0 && directions[0] == direction_t::forward ) {
        //
        // n_domain
        size_t n_domain = domain.size();
        //
        // adom_both
        vector<at::Tensor> dom_both = domain;
        for(size_t k = 0; k < n_domain; ++k) {
            c10::IntArrayRef shape = domain[k].sizes();
            dom_both.push_back( torch::ones( shape ) );
        }
        vector<adten_t> adom_both = adten_t::start_recording(dom_both);
        //
        // adomain, adom_der
        vector<adten_t> adomain, adom_der;
        for(size_t k = 0; k < n_domain; ++k) {
            adomain.push_back( adom_both[k] );
            adom_der.push_back( adom_both[n_domain + k] );
        }
        //
        // arng_der
        vector<adten_t> avar_all = adfn.forward_var(adomain);
        vector<adten_t> arng_der = adfn.forward_der(adom_der, avar_all);
        //
        // adfn_forward
        std::string name = adfn.get_name() + "_forward";
        adfn_t adfn_forward = adten_t::stop_recording(arng_der, name);
        adfn_forward.set_trace( adfn.get_trace() );
        //
        info.m_for_chkpnt_id = make_chkpnt(
            adfn_forward, dom_both, directions.slice(1)
        );
    }
    //
    // info.m_rev_chkpnt_id
    if( directions.size() > 0 && directions[0] == direction_t::reverse ) {
        //
        // n_domain, n_range
        size_t n_domain = domain.size();
        size_t n_range  = adfn.m_rng_shapes.size();
        //
        // adom_both
        vector<at::Tensor> dom_both = domain;
        for(size_t k = 0; k < n_range; ++k) {
            c10::IntArrayRef shape = adfn.m_rng_shapes[k];
            dom_both.push_back( torch::ones( shape ) );
        }
        vector<adten_t> adom_both = adten_t::start_recording(dom_both);
        //
        // adomain, arng_der
        vector<adten_t> adomain, arng_der;
        for(size_t k = 0; k < n_domain; ++k) {
            adomain.push_back( adom_both[k] );
        }
        for(size_t k = 0; k < n_range; ++k) {
            arng_der.push_back( adom_both[n_domain + k] );
        }
        //
        // arng_der
        vector<adten_t> avar_all = adfn.forward_var(adomain);
        vector<adten_t> adom_der = adfn.reverse_der(arng_der, avar_all);
        //
        // adfn_reverse
        std::string name = adfn.get_name() + "_reverse";
        adfn_t adfn_reverse = adten_t::stop_recording(adom_der, name);
        adfn_reverse.set_trace( adfn.get_trace() );
        //
        info.m_rev_chkpnt_id = make_chkpnt(
            adfn_reverse, dom_both, directions.slice(1)
        );
    }
    //
    // info.adfn, info.m_adfn
    auto [depend_par, depend_var] = adfn.forward_dep();
    dev::move_swap( depend_var, info.m_depend );
    dev::move_swap( adfn,       info.m_adfn );
    //
    // info.m_atom_id
    dev::atom_global_t& atom_global = dev::atom_global_t::singleton();
    info.m_atom_id                  = atom_global.number_atom();
    std::unique_ptr<base_atom_t> base_atom_ptr =
        std::make_unique<dev::derive_chkpnt_t>(info);
    size_t atom_id = atom_global.store(base_atom_ptr);
    //
    // chkpnt_id
    dev::chkpnt_global_t& global = dev::chkpnt_global_t::singleton();
    size_t chkpnt_id             = global.store(info);
    //
    return atom_id;
}
//
// BEGIN_CALL_CHKPNT
vector<adten_t> call_chkpnt(
    size_t                 chkpnt_id ,
    const vector<adten_t>& adomain   )
{   // END_CALL_CHKPNT
    //
    size_t atom_id   = chkpnt_id;
    size_t call_info = chkpnt_id;
    return call_atom(atom_id, adomain, call_info);
}

}  // END_AD_TENSOR_NAMESPACE
