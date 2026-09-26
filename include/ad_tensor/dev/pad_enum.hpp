#pragma once
// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
/*
{xrst_begin pad_enum dev}

The Padding Scoped Enum Type
############################

pad_enum_t
**********
{xrst_literal ,
    BEGIN_PAD_ENUM_T, END_PAD_ENUM_T
}

{xrst_end pad_enum}
*/
// BEGIN_PAD_ENUM_T BEGIN_SORT_THIS_LINE_PLUS_2
namespace ad_tensor { namespace dev { enum struct pad_enum_t : std::size_t {
    circular,
    constant,
    reflect,
    replicate
}; } }
// END_PAD_ENUM_T END_SORT_THIS_LINE_MINUS_2
