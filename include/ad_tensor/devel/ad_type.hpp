#pragma once
// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
/*
{xrst_begin ad_type dev}
{xrst_spell
    enum
}

The Types Corresponding to AD tensors
#####################################

Enum Values
***********
{xrst_literal ,
    BEGIN_SORT, END_SORT
}
{xrst_end ad_type}
*/
// BEGIN_SORT_THIS_LINE_PLUS_3
namespace ad_tensor { namespace devel { enum struct ad_type_t {
    constant,  // AD tensor does not depend on domain parameters or variables
    parameter, // AD tensor does not depend on domain variables
    variable,  // AD tensor depends on the domain variables
    none,      // This does not correspond to an AD tensor
}; } }
// END_SORT_THIS_LINE_MINUS_3
