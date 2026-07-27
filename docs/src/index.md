# SpecializingFactorizations.jl

```@docs
MatrixForm
GENERAL
DIAGONAL
LOWER_TRIANGULAR
UPPER_TRIANGULAR
LOWER_BIDIAGONAL
UPPER_BIDIAGONAL
TRIDIAGONAL
BANDED
SYMMETRIC_POSITIVE_DEFINITE
SYMMETRIC_INDEFINITE
HERMITIAN_INDEFINITE
DetectionResult
detect_form
SpecializedLU
SpecializedLU{T}(n::Integer; kl, ku, symmetric) where {T}
reserve!
matrixform(F::SpecializedLU)
isfactored(F::SpecializedLU)
LinearAlgebra.issuccess(::SpecializedLU)
specializinglu
specializinglu!
LinearAlgebra.ldiv!(::AbstractVecOrMat, ::SpecializedLU, ::AbstractVecOrMat)
QRStatus
QR_UNFACTORED
QR_FULLRANK
QR_DEFICIENT
SpecializedQR
SpecializedQR{T}(m::Integer, n::Integer; deficient, nrhs, kl, ku) where {T}
matrixform(F::SpecializedQR)
structuralform
isfactored(F::SpecializedQR)
LinearAlgebra.issuccess(::SpecializedQR)
specializingqr
specializingqr!
LinearAlgebra.ldiv!(::AbstractVecOrMat, ::SpecializedQR, ::AbstractVecOrMat)
```
