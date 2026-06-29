#pragma once
// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
/*
{xrst_begin move_swap dev}
{xrst_spell
    namespace
    std
    dev
}

Swapping using Moves
####################
{xrst_code cpp} */
namespace ad_tensor { namespace dev {
    template<class Type>
    void move_swap(Type& left, Type& right) {
        Type temp( std::move(left) );
        left  = std::move(right);
        right = std::move(temp);
    }
} }
/* {xrst_code}
{xrst_end move_swap}
*/
