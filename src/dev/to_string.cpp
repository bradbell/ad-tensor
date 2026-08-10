// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
#include <ad_tensor/dev/to_string.hpp>
//
namespace ad_tensor { namespace dev {
    //
    // ad_type
    std::string to_string( ad_type_t ad_type)
    {   switch(ad_type) {
            case ad_type_t::constant:  return "con";
            case ad_type_t::parameter: return "par";
            case ad_type_t::variable:  return "var";
            case ad_type_t::none:      return "none";
            default:
            assert( false && "to_string(ad_type): unexpected ad_type" );
        }
        return "";
    }
    //
    // op_enum
    std::string to_string( op_enum_t op_enum) {
        switch(op_enum) {
            // BEGIN_SORT_THIS_LINE_PLUS_1
            case op_enum_t::add:          return "add";
            case op_enum_t::call:         return "call";
            case op_enum_t::call_result:  return "call_result";
            case op_enum_t::div:          return "div";
            case op_enum_t::dom:          return "dom";
            case op_enum_t::eq:           return "eq";
            case op_enum_t::exp:          return "exp";
            case op_enum_t::ge:           return "ge";
            case op_enum_t::gt:           return "gt";
            case op_enum_t::inverse:      return "inverse";
            case op_enum_t::le:           return "le";
            case op_enum_t::logdet:       return "logdet";
            case op_enum_t::lt:           return "lt";
            case op_enum_t::matmul:       return "matmul";
            case op_enum_t::minus:        return "minus";
            case op_enum_t::mul:          return "mul";
            case op_enum_t::ne:           return "ne";
            case op_enum_t::solve:        return "solve";
            case op_enum_t::sub:          return "sub";
            case op_enum_t::sum:          return "sum";
            case op_enum_t::view:         return "view";
            // END_SORT_THIS_LINE_MINUS_1
            default:
            assert( false && "to_string(op_enum): unexpected op_enum value");
            return "Error in dev::to_string(op_enum)";
        }
    }
    //
    // IntArrayRef
    std::string to_string(const c10::IntArrayRef& shape) {
        std::string res = "(";
        for(size_t i = 0; i < shape.size(); ++i) {
            res += std::to_string( shape[i] );
            if( i + 1 < shape.size() )
                res += ", ";
        }
        res += ")";
        return res;
    }
    //
    // ArrayRef<size_t>
    std::string to_string(const c10::ArrayRef<size_t>& vec) {
        std::string res = "(";
        for(size_t i = 0; i < vec.size(); ++i) {
            res += std::to_string( vec[i] );
            if( i + 1 < vec.size() )
                res += ", ";
        }
        res += ")";
        return res;
    }
    //
    // at::Tensor
    std::string to_string(const at::Tensor& tensor) {
        at::IntArrayRef shape  = tensor.sizes();
        std::string res = to_string(shape);
        //
        int64_t length = tensor.numel();
        //
        // vector_view
        at::Tensor vector_view = tensor.contiguous().view(length);
        res += "[";
        for(int64_t i = 0; i < length; ++i) {
            res += std::to_string( vector_view[i].item<double>() );
            if( i + 1 < length )
                res += ", ";
        }
        res += "]";
        return res;
    }
    //
    // adten_t
    std::string to_string(const adten_t& atensor) {
        return to_string( atensor.at_ten() );
    }
    //
    // vector<bool>
    std::string to_string(const vector<bool>& vec) {
        std::string res = "[ ";
        for(size_t i = 0; i < vec.size(); ++i) {
            if( vec[i] ) {
                res += "T";
            } else {
                res += "F";
            }
            if( i + 1 < vec.size() ) {
                res += ", ";
            }
        }
        res += " ]";
        return res;
    }
    //
    // vector<at::Tensor>
    std::string to_string(const vector<at::Tensor>& vec) {
        std::string res;
        for(size_t i = 0; i < vec.size(); ++i) {
            res += "[" + std::to_string(i) + "] = " + to_string(vec[i]) + "\n";
        }
        return res;
    }
    //
    // vector<adten_t>
    std::string to_string(const vector<adten_t>& vec) {
        std::string res;
        for(size_t i = 0; i < vec.size(); ++i) {
            res += "[" + std::to_string(i) + "] = " + to_string(vec[i]) + "\n";
        }
        return res;
    }
    //
    // sparsity_t
    std::string to_string(const sparsity_t& depend) {
        //
        std::string res;
        res += "[";
        for(size_t k = 0; k < depend.size(); ++k) {
            auto [i, j] = depend[k];
            res += "(" + std::to_string(i) + "," + std::to_string(j) + ")";
            if( k + 1 < depend.size() ) {
                res += ", ";
            }
        }
        res += "]";
        return res;
    }
} }
