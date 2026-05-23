// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
#include <ad_tensor/ad.hpp>
#include <ad_tensor/devel/tape.hpp>

namespace ad_tensor {
    ad_t::ad_t( at::Tensor&& tensor )
    : tape_id_(0)
    , index_(0)
    , tensor_(tensor)
    , ad_type_(ad_type_t::constant)
    {   //
        // tape
        devel::tape_t& tape = devel::this_threads_tape();
        if( tape.recording_ ) {
            tape_id_ = tape.tape_id_;
            index_   = tape.con_.size();
            tape.con_.push_back( tensor );
        }
    }
}
