using SpecializingFactorizations
using LinearAlgebra
using Test

# ===========================================================================
# SpecializedQR — rank-revealing least-squares / minimum-norm solver
# ===========================================================================

# correct-typed random matrix/vector, and an exact-rank-r factor product.
_qrand(::Type{T}, dims...) where {T <: Complex} = T.(complex.(randn(dims...), randn(dims...)))
_qrand(::Type{T}, dims...) where {T <: Real} = T.(randn(dims...))
_lowrank(::Type{T}, m, n, r) where {T} = _qrand(T, m, r) * _qrand(T, r, n)
_reltol(::Type{T}) where {T} = (real(T) === Float32 ? 1.0f-3 : 1.0e-8)

@testset "SpecializedQR (rank-revealing least-squares)" begin

    @testset "full-rank LS matches qr(ColumnNorm) and pinv: $T, $shape" for
        T in (Float64, Float32, ComplexF64, ComplexF32),
            shape in (:square, :over, :under)

        m, n = shape === :square ? (5, 5) : (shape === :over ? (7, 4) : (4, 7))
        A = _qrand(T, m, n)
        b = _qrand(T, m)
        F = specializingqr(A)
        x = F \ b
        @test length(x) == n
        # full COLUMN rank (rank == n) is QR_FULLRANK; an underdetermined system
        # at full ROW rank (rank == m < n) takes the min-norm path → QR_DEFICIENT.
        @test matrixform(F) == (n <= m ? QR_FULLRANK : QR_DEFICIENT)
        @test rank(F) == min(m, n)
        @test issuccess(F)
        @test x ≈ qr(A, ColumnNorm()) \ b rtol = _reltol(T)
        @test x ≈ pinv(A) * b rtol = _reltol(T)
        # the normal equations hold: Aᴴ(Ax-b) ≈ 0
        @test norm(A' * (A * x - b)) < _reltol(T) * max(1, norm(A)^2)
    end

    @testset "rank-deficient min-norm matches pinv: $T, $shape" for
        T in (Float64, Float32, ComplexF64, ComplexF32),
            shape in (:square, :over, :under)

        m, n = shape === :square ? (6, 6) : (shape === :over ? (7, 5) : (5, 8))
        r = 3
        A = _lowrank(T, m, n, r)
        b = _qrand(T, m)
        F = specializingqr(A)              # minnorm = true (default)
        x = F \ b
        @test length(x) == n
        @test matrixform(F) == QR_DEFICIENT
        @test rank(F) == r
        @test issuccess(F)                 # rank deficiency is NOT a failure
        @test x ≈ pinv(A) * b rtol = _reltol(T)
        @test x ≈ qr(A, ColumnNorm()) \ b rtol = _reltol(T)
        # min-norm: no other LS solution has smaller norm (compare to basic)
        xbasic = specializingqr(A; minnorm = false) \ b
        @test norm(A' * (A * xbasic - b)) < _reltol(T) * max(1, norm(A)^2)  # valid LS
        @test norm(x) <= norm(xbasic) + _reltol(T)                          # min-norm ≤ basic
    end

    @testset "singular / zero / rank-0 / empty never throw: $T" for
        T in (Float64, Float32, ComplexF64, ComplexF32)

        for (m, n) in ((3, 3), (2, 4), (4, 2))
            A = zeros(T, m, n)
            b = _qrand(T, m)
            F = specializingqr(A)
            x = F \ b
            @test iszero(x)
            @test length(x) == n
            @test rank(F) == 0
            @test issuccess(F)
            @test x ≈ pinv(A) * b
        end
        # empty inputs
        let F = specializingqr(zeros(T, 0, 0))
            @test length(F \ zeros(T, 0)) == 0
            @test rank(F) == 0
        end
        let F = specializingqr(zeros(T, 0, 3))     # 0×3: x has length 3, all zero
            x = F \ zeros(T, 0)
            @test length(x) == 3 && iszero(x)
        end
    end

    @testset "multiple right-hand sides: $T" for T in (Float64, ComplexF64)
        A = _lowrank(T, 7, 5, 3)
        B = _qrand(T, 7, 4)
        F = specializingqr(A)
        X = F \ B
        @test size(X) == (5, 4)
        @test X ≈ pinv(A) * B rtol = _reltol(T)
        # column-wise consistency with single-RHS solves
        for c in 1:4
            @test X[:, c] ≈ F \ B[:, c] rtol = _reltol(T)
        end
    end

    @testset "type stability — one concrete type for every shape/rank" begin
        for T in (Float64, ComplexF64)
            for A in (_qrand(T, 5, 5), _qrand(T, 7, 4), _qrand(T, 4, 7), _lowrank(T, 6, 6, 2))
                @test (@inferred specializingqr(A)) isa SpecializedQR{T, real(T)}
            end
            A = _qrand(T, 7, 4); b = _qrand(T, 7); F = specializingqr(A)
            x = Vector{T}(undef, 4)
            @test (@inferred ldiv!(x, F, b)) === x
            @test (@inferred F \ b) isa Vector{T}
        end
    end

    @testset "zero allocations after setup (warm solve + refactor): $T" for
        T in (Float64, Float32, ComplexF64, ComplexF32)

        @noinline solv(x, F, b) = (ldiv!(x, F, b); @allocated ldiv!(x, F, b))
        @noinline refac(F, A) = (specializingqr!(F, A); @allocated specializingqr!(F, A))
        m, n = 8, 5
        # full column rank
        let A = _qrand(T, m, n), b = _qrand(T, m)
            F = SpecializedQR{T}(m, n)
            specializingqr!(F, A)
            x = Vector{T}(undef, n)
            @test matrixform(F) == QR_FULLRANK
            @test solv(x, F, b) == 0
            @test refac(F, A) == 0
        end
        # rank-deficient (min-norm path: tzrzf/ormrz), buffers reserved upfront
        let A = _lowrank(T, m, n, 2), b = _qrand(T, m)
            F = SpecializedQR{T}(m, n; deficient = true)
            specializingqr!(F, A)
            x = Vector{T}(undef, n)
            @test matrixform(F) == QR_DEFICIENT
            @test solv(x, F, b) == 0
            @test refac(F, A) == 0
        end
    end

    @testset "reserve! makes smaller subsequent solves 0-alloc" begin
        @noinline solv(x, F, b) = (ldiv!(x, F, b); @allocated ldiv!(x, F, b))
        F = SpecializedQR{Float64}()
        reserve!(F, 64, 32; deficient = true, nrhs = 1)
        for (m, n, r) in ((64, 32, 32), (40, 20, 8), (50, 25, 25))
            A = r == n ? _qrand(Float64, m, n) : _lowrank(Float64, m, n, r)
            b = randn(m); x = Vector{Float64}(undef, n)
            specializingqr!(F, A)
            @test solv(x, F, b) == 0
            @test norm(A' * (A * (F \ b) - b)) < 1.0e-7 * max(1, norm(A)^2)
        end
    end

    @testset "rtol keyword controls revealed rank" begin
        # A matrix with one deliberately tiny (but nonzero) singular value.
        U, _ = qr(randn(6, 6)); V, _ = qr(randn(6, 6))
        s = [1.0, 0.5, 0.25, 0.1, 1.0e-9, 1.0e-12]
        A = Matrix(U) * Diagonal(s) * Matrix(V)'
        @test rank(specializingqr(A; rtol = 1.0e-6)) == 4   # drops the 1e-9 and 1e-12
        @test rank(specializingqr(A; rtol = 1.0e-10)) == 5  # keeps 1e-9, drops 1e-12
        @test rank(specializingqr(A; rtol = 1.0e-14)) == 6  # keeps all
    end

    @testset "agreement with Base \\ for tall full-rank (LS)" begin
        A = randn(20, 8); b = randn(20)
        @test specializingqr(A) \ b ≈ A \ b rtol = 1.0e-8
        Ac = randn(ComplexF64, 15, 6); bc = randn(ComplexF64, 15)
        @test specializingqr(Ac) \ bc ≈ Ac \ bc rtol = 1.0e-8
    end

    @testset "generic (non-BLAS) element type: BigFloat" begin
        # rank-deficient: Julia's generic QRPivoted \\ blows up here (≈4e76);
        # our rank-truncated fallback returns a valid LS solution and never throws.
        let A = BigFloat[1 1; 1 1], b = BigFloat[2, 3]
            F = specializingqr(A)
            x = F \ b
            @test rank(F) == 1
            @test norm(A' * (A * x - b)) < 1.0e-60
            @test all(isfinite, x)
        end
        # full-rank overdetermined: matches the exact reference
        let A = BigFloat.(randn(6, 3)), b = BigFloat.(randn(6))
            @test norm(specializingqr(A) \ b - (A \ b)) < 1.0e-60
        end
        # zero matrix: zeros, no throw
        let F = specializingqr(zeros(BigFloat, 3, 3))
            @test iszero(F \ BigFloat.(randn(3)))
            @test rank(F) == 0
        end
    end

    @testset "integer / rational eltypes promote (QR needs sqrt)" begin
        Ai = [2 1 0; 1 3 1; 0 1 2]; bi = [1, 2, 3]
        F = specializingqr(Ai)
        @test eltype(F) === Float64
        @test F \ bi ≈ Float64.(Ai) \ Float64.(bi)
        Ar = Rational{Int}[1 2; 2 4]; br = Rational{Int}[1, 2]
        Fr = specializingqr(Ar)        # promotes to Float64 (no exact rational QR)
        @test eltype(Fr) === Float64
        @test rank(Fr) == 1
        @test norm(Float64.(Ar)' * (Float64.(Ar) * (Fr \ br) - Float64.(br))) < 1.0e-10
    end

    @testset "fallback = false leaves the QR to the host" begin
        A = randn(6, 4)
        F = specializingqr(A; fallback = false)
        @test !isfactored(F)
        @test matrixform(F) == QR_UNFACTORED
        @test_throws ArgumentError ldiv!(zeros(4), F, ones(6))
    end

    @testset "dimension mismatches throw" begin
        A = randn(6, 4); F = specializingqr(A)
        @test_throws DimensionMismatch ldiv!(zeros(4), F, ones(5))   # wrong rhs rows
        @test_throws DimensionMismatch ldiv!(zeros(3), F, ones(6))   # wrong x rows
        @test_throws DimensionMismatch ldiv!(F, ones(6))             # 2-arg needs square
    end

    # ----- structure specialization (detect_form reused by the QR solver) -----

    # A well-conditioned, diagonally-dominant instance of each structured form.
    function _struct_mat(form::MatrixForm, ::Type{T}, n::Int) where {T}
        dom() = T <: Complex ? T(n + 3, 0) : T(n + 3)
        off() = T <: Complex ? T(0.1, 0.1) : T(0.1)
        if form == DIAGONAL
            return Matrix(Diagonal(T[dom() for _ in 1:n]))
        elseif form == UPPER_TRIANGULAR
            return T[i < j ? off() : (i == j ? dom() : zero(T)) for i in 1:n, j in 1:n]
        elseif form == LOWER_TRIANGULAR
            return T[i > j ? off() : (i == j ? dom() : zero(T)) for i in 1:n, j in 1:n]
        elseif form == UPPER_BIDIAGONAL
            return Matrix(Bidiagonal(T[dom() for _ in 1:n], T[off() for _ in 1:(n - 1)], :U))
        elseif form == LOWER_BIDIAGONAL
            return Matrix(Bidiagonal(T[dom() for _ in 1:n], T[off() for _ in 1:(n - 1)], :L))
        elseif form == TRIDIAGONAL
            return Matrix(Tridiagonal(T[off() for _ in 1:(n - 1)], T[dom() for _ in 1:n], T[off() for _ in 1:(n - 1)]))
        else # BANDED (pentadiagonal)
            return diagm(
                0 => T[dom() for _ in 1:n],
                1 => T[off() for _ in 1:(n - 1)], -1 => T[off() for _ in 1:(n - 1)],
                2 => T[off() for _ in 1:(n - 2)], -2 => T[off() for _ in 1:(n - 2)],
            )
        end
    end

    @testset "structured forms match pinv & geqp3: $T, $form" for
        T in (Float64, Float32, ComplexF64, ComplexF32),
            form in (
                DIAGONAL, UPPER_TRIANGULAR, LOWER_TRIANGULAR,
                UPPER_BIDIAGONAL, LOWER_BIDIAGONAL, TRIDIAGONAL, BANDED,
            )

        n = 9
        A = _struct_mat(form, T, n)
        b = _qrand(T, n)
        F = specializingqr(A)
        x = F \ b
        @test structuralform(F) == form          # took the structured fast path
        @test rank(F) == n
        @test matrixform(F) == QR_FULLRANK
        @test issuccess(F)
        @test x ≈ pinv(A) * b rtol = _reltol(T)
        @test x ≈ qr(A, ColumnNorm()) \ b rtol = _reltol(T)
        # indistinguishable from the dense rank-revealing path:
        Fg = specializingqr(A; detect_structure = false)
        @test structuralform(Fg) == GENERAL
        @test rank(F) == rank(Fg)
        @test x ≈ Fg \ b rtol = _reltol(T)
        # multi-RHS
        B = _qrand(T, n, 3)
        @test F \ B ≈ pinv(A) * B rtol = _reltol(T)
    end

    @testset "DIAGONAL rank-revealing + singular (pure structured): $T" for
        T in (Float64, Float32, ComplexF64, ComplexF32)

        # exact zeros and a sub-tolerance entry ⇒ rank deficiency, no fallback
        d = T[2, 0, 3, 0, 4]
        d[5] = T(maximum(abs, d)) * (5 * eps(real(T)) / 4)  # sub-tolerance ⇒ dropped
        A = Matrix(Diagonal(d))
        b = _qrand(T, 5)
        F = specializingqr(A)
        x = F \ b
        @test structuralform(F) == DIAGONAL          # still structured (no fallback)
        @test rank(F) == 2
        @test issuccess(F)                           # deficiency is success
        @test all(isfinite, x)
        @test x ≈ pinv(A) * b rtol = _reltol(T)
        @test x ≈ qr(A, ColumnNorm()) \ b rtol = _reltol(T)
        # all-zero diagonal ⇒ rank 0, zero solution, no throw
        let Z = zeros(T, 4, 4), F0 = specializingqr(Z)
            @test rank(F0) == 0
            @test structuralform(F0) == DIAGONAL
            @test iszero(F0 \ _qrand(T, 4))
        end
        # the rank matches geqp3 across a randomized stress of zeros/tiny entries
        for _ in 1:50
            dd = _qrand(T, 8)
            for k in 1:8
                rand() < 0.3 && (dd[k] = zero(T))
            end
            AA = Matrix(Diagonal(dd))
            @test rank(specializingqr(AA)) == rank(specializingqr(AA; detect_structure = false))
        end
    end

    @testset "ill-conditioned / singular structured fall back to geqp3 (contract)" begin
        # graded ill-conditioned upper-triangular: gate fails ⇒ geqp3, and the
        # rank + solution stay identical to the dense rank-revealing reference.
        n = 12
        A = triu(ones(n, n))
        for i in 1:n
            A[i, i] = 10.0^(-1.4 * (i - 1))            # cond ≈ 1e15, past the rtol boundary
        end
        b = randn(n)
        F = specializingqr(A)
        Fg = specializingqr(A; detect_structure = false)
        @test structuralform(F) == GENERAL            # gate failed ⇒ fell back
        @test rank(F) == rank(Fg)                     # same revealed rank as geqp3
        @test F \ b ≈ Fg \ b rtol = 1.0e-8            # same solution as geqp3
        @test F \ b ≈ pinv(A) * b rtol = 1.0e-7
        @test all(isfinite, F \ b)

        # exact zero on a triangular diagonal: a plain trtrs would THROW; we must
        # fall back to geqp3 and return the finite min-norm solution.
        let U = [2.0 1.0 3.0; 0.0 0.0 4.0; 0.0 0.0 5.0], bz = [1.0, 2.0, 3.0]
            Fz = specializingqr(U)
            @test structuralform(Fz) == GENERAL
            @test all(isfinite, Fz \ bz)
            @test Fz \ bz ≈ pinv(U) * bz rtol = 1.0e-9
            @test rank(Fz) == 2
        end

        # rank-deficient tridiagonal (two identical rows ⇒ rank 2 of 3): the
        # hand-rolled gttrf would otherwise divide by a zero pivot; must fall back.
        let A = Matrix(Tridiagonal([1.0, 1.0], [0.0, 0.0, 0.0], [1.0, 1.0])), bt = [1.0, 2.0, 3.0]
            Ft = specializingqr(A)
            @test structuralform(Ft) == GENERAL
            @test all(isfinite, Ft \ bt)
            @test Ft \ bt ≈ pinv(A) * bt rtol = 1.0e-9
            @test rank(Ft) == rank(specializingqr(A; detect_structure = false))
        end
        # ill-conditioned banded full-rank-but-past-rtol: gate must route to geqp3.
        let nb = 30
            A = diagm(
                0 => 10.0 .^ range(0, -16; length = nb),
                1 => fill(1.0, nb - 1), -1 => fill(1.0, nb - 1),
                2 => fill(1.0, nb - 2), -2 => fill(1.0, nb - 2),
            )
            bb = randn(nb)
            Fb = specializingqr(A)
            Fbg = specializingqr(A; detect_structure = false)
            @test rank(Fb) == rank(Fbg)
            @test Fb \ bb ≈ Fbg \ bb rtol = 1.0e-7
            @test all(isfinite, Fb \ bb)
        end
    end

    @testset "structured paths: 0 allocations (warm solve + refactor): $T" for
        T in (Float64, Float32, ComplexF64, ComplexF32)

        @noinline solv(x, F, b) = (ldiv!(x, F, b); @allocated ldiv!(x, F, b))
        @noinline refac(F, A) = (specializingqr!(F, A); @allocated specializingqr!(F, A))
        n = 10
        for form in (
                DIAGONAL, UPPER_TRIANGULAR, LOWER_TRIANGULAR,
                UPPER_BIDIAGONAL, LOWER_BIDIAGONAL, TRIDIAGONAL, BANDED,
            )
            A = _struct_mat(form, T, n)
            b = _qrand(T, n)
            # reserve the band buffer for the BANDED path (kl=ku=2 pentadiagonal)
            F = form == BANDED ? SpecializedQR{T}(n, n; kl = 2, ku = 2) : SpecializedQR{T}(n, n)
            specializingqr!(F, A)
            x = Vector{T}(undef, n)
            @test structuralform(F) == form
            @test solv(x, F, b) == 0
            @test refac(F, A) == 0
        end
    end

    @testset "structured paths: type stability and escape hatches" begin
        for T in (Float64, ComplexF64)
            for form in (DIAGONAL, UPPER_TRIANGULAR, LOWER_TRIANGULAR)
                A = _struct_mat(form, T, 6)
                @test (@inferred specializingqr(A)) isa SpecializedQR{T, real(T)}
                F = specializingqr(A)
                x = Vector{T}(undef, 6); b = _qrand(T, 6)
                @test (@inferred ldiv!(x, F, b)) === x
            end
        end
        # detect_structure = false forces the dense path even for a diagonal A,
        # and returns the identical solution.
        let A = Matrix(Diagonal(randn(6) .+ 3.0)), b = randn(6)
            @test structuralform(specializingqr(A; detect_structure = false)) == GENERAL
            @test specializingqr(A) \ b ≈ specializingqr(A; detect_structure = false) \ b
        end
        # rectangular input must NOT call detect_form (it is square-only / throws)
        @test structuralform(specializingqr(randn(7, 4))) == GENERAL
        @test structuralform(specializingqr(randn(4, 7))) == GENERAL
    end

    @testset "band gate: Varah early-accept and laic1 fallback agree with geqp3: $T" for
        T in (Float64, Float32, ComplexF64, ComplexF32)

        rt = _reltol(T)
        n = 16
        # strongly diagonally dominant ⇒ the O(n)/O(n·b) Varah early-accept fires
        # (no laic1 sweep); NOT diagonally dominant but still full rank ⇒ Varah
        # declines and the O(n²) laic1 gate accepts. Both must equal geqp3/pinv.
        for (kl, ku) in ((1, 1), (2, 2))
            dl = T[_qrand(T, 1)[1] for _ in 1:(n - 1)]
            dom = Matrix(Tridiagonal(dl .* T(0.2), T[_qrand(T, 1)[1] + T(8) for _ in 1:n], dl .* T(0.2)))
            if kl == 2
                dom = dom + diagm(2 => fill(T(0.1), n - 2), -2 => fill(T(0.1), n - 2))
            end
            # off-diagonals larger than the diagonal ⇒ not diagonally dominant
            nondom = Matrix(
                Tridiagonal(
                    T[_qrand(T, 1)[1] + T(3) for _ in 1:(n - 1)],
                    T[_qrand(T, 1)[1] * T(0.5) for _ in 1:n],
                    T[_qrand(T, 1)[1] + T(3) for _ in 1:(n - 1)],
                )
            )
            for A in (dom, nondom)
                b = _qrand(T, n)
                F = specializingqr(A)
                Fg = specializingqr(A; detect_structure = false)
                @test structuralform(F) in (TRIDIAGONAL, BANDED)
                @test rank(F) == rank(Fg)                 # Varah/laic1 path == geqp3 rank
                @test F \ b ≈ pinv(A) * b rtol = rt
                @test F \ b ≈ Fg \ b rtol = rt
            end
        end
        # a barely-non-dominant-but-rank-deficient tridiagonal must still fall to
        # geqp3 (Varah declines; laic1/gttrf detect the deficiency): never a
        # spurious full-rank Varah accept.
        let A = Matrix(Tridiagonal([1.0, 1.0], [0.0, 0.0, 0.0], [1.0, 1.0])), bz = [1.0, 2.0, 3.0]
            F = specializingqr(A)
            @test structuralform(F) == GENERAL
            @test rank(F) == 2
            @test F \ bz ≈ pinv(A) * bz rtol = 1.0e-9
        end
    end
end
