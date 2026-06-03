// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
#include <ad_tensor/dev/to_string.hpp>
//
namespace ad_tensor { namespace dev {
    std::string to_string( ad_type_t ad_type)
    {   switch(ad_type) {
            case ad_type_t::constant:  return "constant";
            case ad_type_t::parameter: return "parameter";
            case ad_type_t::variable:  return "variable";
            case ad_type_t::none:      return "none";
            default:
            assert( false && "to_string(ad_type): unexpected ad_type" );
        }
        return "";
    }
    std::string to_string( op_enum_t op_enum) {
        switch(op_enum) {
            // BEGIN_SORT_THIS_LINE_PLUS_1
            case op_enum_t::add:    return "add";
            case op_enum_t::div:    return "div";
            case op_enum_t::dom:    return "dom";
            case op_enum_t::minus:  return "minus";
            case op_enum_t::mul:    return "mul";
            case op_enum_t::sub:    return "sub";
            case op_enum_t::sum:    return "sum";
            // END_SORT_THIS_LINE_MINUS_1
            default:
            assert( false && "to_string(op_enum): unexpected op_enum value");
        }
    }
    std::string to_string(const at::Tensor& tensor) {
        at::IntArrayRef shape  = tensor.sizes();
        std::string res = "(";
        for(size_t i = 0; i < shape.size(); ++i) {
            res += std::to_string( shape[i] );
            if( i + 1 < shape.size() )
                res += ", ";
        }
        res += ")";
        //
        size_t length = tensor.numel();
        res += "[";
        for(size_t i = 0; i < length; ++i) {
            res += std::to_string( tensor.data_ptr<float>()[i] );
            if( i + 1 < length )
                res += ", ";
        }
        res += "]";
        return res;
    }
} }
