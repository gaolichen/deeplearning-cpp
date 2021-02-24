#pragma once
#include "common.h"

typedef data_t (*dataTransformFun)(data_t);
typedef Eigen::Map<Eigen::ArrayXi> IndexType;

class DataUtil {
public:
    static void appendColumnProduct(Matrix& mat, std::vector<int> cols);
    
    static void appendCustomColumn(Matrix& mat, int column, dataTransformFun);
    
    static std::vector<data_t> smooth(const std::vector<data_t>& data, int range = 10);
    
    static std::vector<int> randomIndex(int range);
    
    static IndexType randomSubIndex(int range, int setSize);
    
    static Matrix randomRowShuffle(const Matrix& data);
    
    static PermutationMatrix randomPermutation(int size);
    
    static std::vector<data_t> eigenVectorToSTL(const Vector& vec, bool smooth = false);
    
    static std::vector<data_t> eigenArrayToSTL(const Array& vec, bool smooth = false);
    
    static Matrix zScoreNormalize(const Matrix& mat);
    
    static Array colWiseStd(const Matrix& mat);
    
    static Vector randomDropoutVector(int size, data_t dropRate, const std::vector<int>& noDropPositions);
};

class CSVData {
    std::vector<std::string> _headers;
    std::vector<data_t> _innerData;
    Matrix _data;
public:
    void read(std::string file, bool hasHeader = false);
    
    Matrix& data() {
        return _data;
    }
    
    const Matrix& filter(std::vector<std::string> headers) {
        return _data;
    }
    
    std::string header(int i) {
        return _headers[i];
    }
    
    int headerIndex(std::string header) {
        auto it = std::find(_headers.begin(), _headers.end(), header);
        if (it != _headers.end()) {
            return it - _headers.begin();
        } else {
            return -1;
        }
    }
    
    const std::vector<std::string> headers() {
        return _headers;
    }
};

template<class ArgType, class RowIndexType>
class indexing_functor {
  const ArgType &m_arg;
  const RowIndexType &m_rowIndices;
public:
  typedef Eigen::Matrix<typename ArgType::Scalar,
                 RowIndexType::SizeAtCompileTime,
                 RowIndexType::SizeAtCompileTime,
                 ArgType::Flags&Eigen::RowMajorBit?Eigen::RowMajor:Eigen::ColMajor,
                 RowIndexType::MaxSizeAtCompileTime,
                 RowIndexType::MaxSizeAtCompileTime> MatrixType;
 
  indexing_functor(const ArgType& arg, const RowIndexType& row_indices)
    : m_arg(arg), m_rowIndices(row_indices)
  {}
 
  const typename ArgType::Scalar& operator() (Eigen::Index row, Eigen::Index col) const {
    return m_arg(m_rowIndices[row], col);
  }
};

template <class ArgType, class RowIndexType>
Eigen::CwiseNullaryOp<indexing_functor<ArgType,RowIndexType>, typename indexing_functor<ArgType,RowIndexType>::MatrixType>
indexing(const Eigen::MatrixBase<ArgType>& arg, const RowIndexType& row_indices)
{
  typedef indexing_functor<ArgType,RowIndexType> Func;
  typedef typename Func::MatrixType MatrixType;
  return MatrixType::NullaryExpr(row_indices.size(), arg.cols(), Func(arg.derived(), row_indices));
}

