#pragma once
// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
#include <ad_tensor/ad_type.hpp>
#include <ad_tensor/dev/op_enum.hpp>
#include <ad_tensor/dev/agraph.hpp>
#include <ad_tensor/dev/vec_sets.hpp>
namespace ad_tensor { namespace dev {
    void call_op_depend(
        size_t          op_index    ,
        const agraph_t& agraph      ,
        ad_type_t       domain_type ,
        vec_sets_t&     vec_sets
    );
} }
