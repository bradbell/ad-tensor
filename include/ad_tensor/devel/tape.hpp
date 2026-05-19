#pragma once
// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
/*
{xrst_begin tape dev}

Tape that Records Operation Sequences
#####################################
{xrst_literal ,
    BEGIN_TAPE_T, END_TAPE_T
}

con
***
is the vector of constant values.

par
***
is the acyclic graph for the operation sequence that determines
the dependent parameters.

var
***
is the acyclic graph for the operation sequence that determines
the dependent variables.

tape_id
*******
If *recording* is true, this is the unique identifier for the recording.

recording
*********
is true (false) if this tape is currently recording (not recording)
AD operations.

default constructor
*******************
The default constructor creates an empty tape with tape_id zero and
recording false.

is_empty
********
returns true if the constant vector, the parameter graph,
and the variable graph are empty.

this_threads_tape
*****************
is the tape used to record AD operations on this thread.

{xrst_end tape}
*/
#include <vector>
#include <torch/torch.h>
//
#include <ad_tensor/devel/agraph.hpp>
#include <ad_tensor/ad.hpp>
//
// record_t
namespace ad_tensor { class record_t; }
//
// BEGIN_TAPE_T
namespace ad_tensor { namespace devel { class tape_t {
    friend class ad_tensor::record_t;
private:
    std::vector<torch::Tensor> con_;
    agraph_t                   par_;
    agraph_t                   var_;
    size_t                     tape_id_;
    bool                       recording_;
public:
    //
    // default constructor
    tape_t() : con_(), par_(), var_(), tape_id_(0), recording_(false)
    { }
    //
    // tape_id, recording
    size_t tape_id(void) const noexcept { return tape_id_; }
    size_t recording(void) const noexcept { return recording_; }
    //
    // is_empty
    bool is_empty(void) const {
        return con_.empty() && par_.is_empty() && var_.is_empty();
    }
}; } }
//
// this_threads_tape
namespace ad_tensor { namespace devel {
    thread_local tape_t this_threads_tape;
} }
// END_TAPE_T
