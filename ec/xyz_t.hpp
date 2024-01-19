// hessian implementation


    __host__ __device__ void add(const xyzz_t& p2)
    {
        if (p2.is_inf()) {
            return;
        } else if (is_inf()) {
            *this = p2;
            return;
        }

#ifdef __CUDA_ARCH__
        xyzz_t p31 = *this;
#else
        xyzz_t& p31 = *this;
#endif
        field_t U, S, P, R;

        U = p31.X * p2.ZZ;          /* U1 = X1*ZZ2 */
        S = p31.Y * p2.ZZZ;         /* S1 = Y1*ZZZ2 */
        P = p2.X * p31.ZZ;          /* U2 = X2*ZZ1 */
        R = p2.Y * p31.ZZZ;         /* S2 = Y2*ZZZ1 */
        P -= U;                     /* P = U2-U1 */
        R -= S;                     /* R = S2-S1 */

        if (!P.is_zero()) {         /* X1!=X2 */
            field_t PP;             /* add |p1| and |p2| */

            PP = P^2;               /* PP = P^2 */
#define PPP P
            PPP = P * PP;           /* PPP = P*PP */
            p31.ZZ *= PP;           /* ZZ3 = ZZ1*ZZ2*PP */
            p31.ZZZ *= PPP;         /* ZZZ3 = ZZZ1*ZZZ2*PPP */
#define Q PP
            Q = U * PP;             /* Q = U1*PP */
            p31.X = R^2;            /* R^2 */
            p31.X -= PPP;           /* R^2-PPP */
            p31.X -= Q;
            p31.X -= Q;             /* X3 = R^2-PPP-2*Q */
            Q -= p31.X;
            Q *= R;                 /* R*(Q-X3) */
            p31.Y = S * PPP;        /* S1*PPP */
            p31.Y = Q - p31.Y;      /* Y3 = R*(Q-X3)-S1*PPP */
            p31.ZZ *= p2.ZZ;        /* ZZ1*ZZ2 */
            p31.ZZZ *= p2.ZZZ;      /* ZZZ1*ZZZ2 */
#undef PPP
#undef Q
        } else if (R.is_zero()) {   /* X1==X2 && Y1==Y2 */
            field_t M;              /* double |p1| */

            U = p31.Y + p31.Y;      /* U = 2*Y1 */
#define V P
#define W R
            V = U^2;                /* V = U^2 */
            W = U * V;              /* W = U*V */
            S = p31.X * V;          /* S = X1*V */
            M = p31.X^2;
            M = M + M + M;          /* M = 3*X1^2[+a*ZZ1^2] */
            p31.X = M^2;
            p31.X -= S;
            p31.X -= S;             /* X3 = M^2-2*S */
            p31.Y *= W;             /* W*Y1 */
            S -= p31.X;
            S *= M;                 /* M*(S-X3) */
            p31.Y = S - p31.Y;      /* Y3 = M*(S-X3)-W*Y1 */
            p31.ZZ *= V;            /* ZZ3 = V*ZZ1 */
            p31.ZZZ *= W;           /* ZZZ3 = W*ZZZ1 */
#undef V
#undef W
        } else {                    /* X1==X2 && Y1==-Y2 */\
            p31.inf();              /* set |p3| to infinity */\
        }
#ifdef __CUDA_ARCH__
        *this = p31;
#endif
    }

#ifdef __CUDA_ARCH__
    __device__ void uadd(const xyzz_t& p2)
    {
        xyzz_t p31 = *this;

        if (p2.is_inf()) {
            return;
        } else if (p31.is_inf()) {
            *this = p2;
            return;
        }

        field_t A, B, U, S, P, R, PP;
        int pc = -1;
        bool done = false, dbl = false, inf = false;

        A = p31.Y;
        B = p2.ZZZ;
        #pragma unroll 1
        do {
            A = A * B;
            switch (++pc) {
            case 0:
                S = A;                  /* S1 = Y1*ZZZ2 */
                A = p2.Y;
                B = p31.ZZZ;
                break;
            case 1:                     /* S2 = Y2*ZZZ1 */
                R = A - S;              /* R = S2-S1 */
                A = p31.X;
                B = p2.ZZ;
                break;
            case 2:
                U = A;                  /* U1 = X1*ZZ2 */
                A = p2.X;
                B = p31.ZZ;
                break;
            case 3:                     /* U2 = X2*ZZ1 */
                A = A - U;              /* P = U2-U1 */
                inf = A.is_zero();      /* X1==X2, not add |p1| and |p2| */
                dbl = R.is_zero() & inf;
                if (dbl) {              /* X1==X2 && Y1==Y2, double |p2| */
                    A = p2.Y<<1;        /* U = 2*Y1 */
                    inf = false;        /* don't set |p3| to infinity */
                }
                B = A;
                break;
            case 4:
                PP = A;                 /* PP = P^2 */
                break;
            case 5:
#define PPP P
                PPP = A;                /* PPP = P*PP */
                B = field_t::csel(field_t::one(), p31.ZZZ, dbl);
                break;
            case 6:                     /* ZZZ1*PPP */
                B = czero(p2.ZZZ, inf);
                break;
            case 7:
                p31.ZZZ = A;            /* ZZZ3 = ZZZ1*ZZZ2*PPP */
                A = field_t::csel(field_t::one(), p31.ZZ, dbl);
                B = czero(p2.ZZ, inf);
                break;
            case 8:                     /* ZZ1*ZZ2 */
                B = PP;
                break;
            case 9:
                p31.ZZ = A;             /* ZZ3 = ZZ1*ZZ2*PP */
                A = field_t::csel(p2.X, U, dbl);
                break;
            case 10:
#define Q PP
                Q = A;                  /* Q = U1*PP */
                A = field_t::csel(p2.Y, S, dbl);
                B = PPP;
                break;
            case 11:
                p31.Y = A;              /* S1*PPP */
                A = R;
                B = R;
                break;
            case 12:                    /* R^2 */
                p31.X = A - PPP;        /* R^2-PPP */
                p31.X -= Q;
                p31.X -= Q;             /* X3 = R^2-PPP-2*Q */
                A = Q - p31.X;
                break;
            case 13:                    /* R*(Q-X3) */
                //p31.Y = A - p31.Y;    /* Y3 = R*(Q-X3)-S1*PPP */
                if (dbl) {
                    A = p2.X;
                    B = p2.X;
                } else {
                    done = true;
                }
                break;
#undef PPP
#undef Q
            /*** double |p2|, V*X1, W*Y1, ZZZ3 and ZZ3 are already calculated ***/
#define S PP
            case 14:
                A = A + A + A;          /* M = 3*X1^2[+a*ZZ1^2] */
                B = A;
                break;
            case 15:
                p31.X = A - S - S;      /* X3 = M^2-2*S */
                A = S - p31.X;
                break;
            case 16:                    /* M*(S-X3) */
                //p31.Y = A - p31.Y;    /* Y3 = M*(S-X3)-W*Y1 */
                done = true;
                break;
#undef S
            }
        } while (!done);
        p31.Y = A - p31.Y;              /* Y3 = R*(Q-X3)-S1*PPP */

        *this = p31;
    }
#else
    inline void uadd(const xyzz_t& p2) { add(p2); }
#endif