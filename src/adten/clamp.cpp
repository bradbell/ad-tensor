// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
/*
-------------------------------------------------------------------------------
{xrst_begin adten_clamp usr}
{xrst_spell
}

Element-wise Bounding
#####################

Syntax
******
{xrst_code cpp}
    clamped = input.clamp(lower, upper)
{xrst_code}

Prototype
*********
{xrst_literal ,
    BEGIN_CLAMP, END_CLAMP
}

lower
*****
is the lower limit for the bounding.

upper
*****
is the upper limit for the bounding.

clamped
*******
is the result of the element-wise bounding.
If an element of input is
less than lower (greater than upper) [between lower and upper]
the corresponding element of clamped is
equal to lower (equal to upper) [equal to input].


Example
*******
{xrst_literal ,
    examples/adten/clamp.cpp
    BEGIN_CPP, END_CPP
}
{xrst_end adten_clamp}
-------------------------------------------------------------------------------
*/
#include <torch/torch.h>
#include <ad_tensor/adten.hpp>
#include <ad_tensor/dev/tape.hpp>
#include <ad_tensor/dev/user_assert.hpp>

namespace ad_tensor { // Begin ad_tensor::dev

// BEGIN_CLAMP
adten_t adten_t::clamp(const adten_t& lower, const adten_t& upper) const
{   // END_CLAMP
    const adten_t& input = *this;
    return input.maximum(lower).minimum(upper);
}
// ---------------------------------------------------------------------------
} // End ad_tensor
