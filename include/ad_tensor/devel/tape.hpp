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

swap
****
Exchange the contents between two tapes.

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
//
// BEGIN_TAPE_T
namespace ad_tensor { namespace devel { struct tape_t {
    std::vector<torch::Tensor> con;
    agraph_t                   par;
    agraph_t                   var;
    size_t                     tape_id;
    bool                       recording;
    //
    // default constructor
    tape_t() : con(), par(), var(), tape_id(0), recording(false)
    { }
    //
    // swap
    void swap(tape_t& other)
    {   con.swap(other.con);
        par.swap(other.par);
        var.swap(other.var);
        std::swap(tape_id, other.tape_id);
        std::swap(recording, other.recording);
    }
    //
    // is_empty
    bool is_empty(void) {
        return con.empty() && par.is_empty() && var.is_empty();
    }
}; } }
//
// this_threads_tape
namespace ad_tensor { namespace devel { 
    thread_local tape_t this_threads_tape;
} }
// END_TAPE_T


