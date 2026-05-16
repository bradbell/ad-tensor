#pragma once
// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
/*
{xrst_begin tape dev}

Tape that Records Operation Sequences
#####################################

Fields
******
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
    // is_empty
    bool is_empty(void) {
        return con.empty() && par.is_empty() && var.is_empty();
    }
}; } }
// END_TAPE_T
