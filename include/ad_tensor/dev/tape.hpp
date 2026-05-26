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
the dependent parameter tensors.

var
***
is the acyclic graph for the operation sequence that determines
the dependent variable tensors.

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
is the thread local tape used to record AD operations (on this thread).

{xrst_end tape}
*/
#include <vector>
#include <torch/torch.h>
//
#include <ad_tensor/dev/agraph.hpp>
#include <ad_tensor/ad.hpp>
//
// BEGIN_TAPE_T
namespace ad_tensor { namespace dev { class tape_t {
    friend class ad_tensor::ad_t;
private:
    std::vector<at::Tensor>    m_con;
    agraph_t                   m_par;
    agraph_t                   m_var;
    size_t                     m_tape_id;
    bool                       m_recording;
public:
    //
    // default constructor
    tape_t() : m_con(), m_par(), m_var(), m_tape_id(0), m_recording(false)
    { }
    //
    // tape_id, recording
    size_t tape_id(void) const noexcept { return m_tape_id; }
    size_t recording(void) const noexcept { return m_recording; }
    //
    // is_empty
    bool is_empty(void) const {
        return m_con.empty() && m_par.is_empty() && m_var.is_empty();
    }
}; } }
//
// this_threads_tape
namespace ad_tensor { namespace dev {
    tape_t& this_threads_tape(void);
} }
// END_TAPE_T
