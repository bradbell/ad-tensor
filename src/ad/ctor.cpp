// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
#include <ad_tensor/ad.hpp>
#include <ad_tensor/dev/tape.hpp>
#include <ad_tensor/dev/op_enum.hpp>
#include <ad_tensor/dev/agraph.hpp>
#include <ad_tensor/dev/user_assert.hpp>
//
namespace ad_tensor { // BEGIN_NAMESPACE_AD_TENSOR
// ----------------------------------------------------------------------------
ad_t::ad_t( at::Tensor&& tensor )
: m_tape_id(0)
, m_index(0)
, m_tensor(tensor)
, m_ad_type(ad_type_t::constant)
{   //
    // tape
    dev::tape_t& tape = dev::this_threads_tape();
    if( tape.m_recording ) {
        m_tape_id = tape.m_tape_id;
        m_index   = tape.m_con.size();
        tape.m_con.push_back( tensor.clone() );
    }
}
// ---------------------------------------------------------------------------
} // END_NAMESPACE_AD_TENSOR
