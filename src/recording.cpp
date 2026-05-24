// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
#include <torch/torch.h>
//
#include <ad_tensor/ad.hpp>
#include <ad_tensor/adfn.hpp>
#include <ad_tensor/ad_type.hpp>
#include <ad_tensor/devel/op_enum.hpp>
#include <ad_tensor/devel/tape.hpp>
#include <ad_tensor/recording.hpp>
//
namespace ad_tensor { // BEGIN_NAMESPACE_AD_TENSOR
/*
------------------------------------------------------------------------------
{xrst_begin_parent recording usr}

The recording Class
###################
This class is special in two ways;
It only has static member functions and its name does not end with ``_t`` .

{xrst_end recording}
------------------------------------------------------------------------------
{xrst_begin recording_stop usr}
{xrst_spell
    adfn
    arange
}

Stop Recording ad_t Operations
##############################
{xrst_literal,
    include/ad_tensor/recording.hpp
    BEGIN_STOP, END_STOP
}

Recording
*********
This thread must have a recording in progress
when ``recording::stop`` is called.
The recording started by calling
:ref:`start_recording-name` .

arange
******
This vector of AD tensors specifies the range for the AD function *adfn* .

adfn
****
The operation sequence that was recording is transferred to this AD function.
The domain parameter and variable tensors for this function are defined
in the corresponding :ref:`start_recording-name` .

Example
*******
{xrst_literal ,
    examples/recording.cpp
    BEGIN_CPP, END_CPP
}

{xrst_end recording_stop}
*/
// stop
adfn_t recording::stop(const std::vector<ad_t>& arange)
{   //
    // tape
    devel::tape_t& tape = devel::this_threads_tape();
    //
    assert( tape.recording() &&
        "recording::stop: this threads tape is not recording"
    );
    assert( ! tape.is_empty() &&
        "recording_stop: a tape that is recording should not be empty"
    );
    //
    // tape
    tape.m_recording = false;
    //
    // adfn
    adfn_t adfn;
    //
    // adfn, tape
    adfn.m_con.swap( tape.m_con );
    adfn.m_par.swap( tape.m_par );
    adfn.m_var.swap( tape.m_var );
    //
    // adfn
    for(size_t i = 0; i < arange.size(); ++i)
    {   adfn.m_rng_index.push_back( arange.at(i).m_index );
        adfn.m_rng_ad_type.push_back( arange.at(i).m_ad_type );
    }
    //
    return adfn;
}
// ---------------------------------------------------------------------------
} // END_NAMESPACE_AD_TENSOR
