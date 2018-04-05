using WarpedImageSeries, BlockRegistration, CoordinateTransformations

using Base.Test

all_nan_eq(a1, a2) = all(map((x,y)->isequal(x,y), a1, a2))

#2D
n = 100
a = rand(100,100,n);
tfms = fill(Translation(0.1,0.2), n);
ws = warpedseries(a, tfms);
@test size(a) == size(ws)
wrpd = warp(a[:,:,1], tfms[1]);
@test all_nan_eq(ws[:,:,1], wrpd[Base.front(indices(a))...])
wrpd = warp(a[:,:,2], tfms[2]);
@test all_nan_eq(ws[:,:,2], wrpd[Base.front(indices(a))...])

tfms2 = fill(Translation(1.1,1.2), n);
ws2 = warpedseries(a, tfms2);
@test size(ws2,1) == size(a,1) - 1
@test size(ws2,2) == size(a,2) - 1

sls = ws2[:,:,2:4];
for (n,i) in enumerate(2:4)
    @test all_nan_eq(ws2[:,:,i], sls[:,:,n])
end
#@time hm = ws2[:,:,1];
#@time hm = ws2[:,:,1:6];

#3D
n = 10
a = rand(10,10,10,n);
tfms = fill(Translation(0.1,0.2,0.15), n);
ws = warpedseries(a, tfms);
@test size(a) == size(ws)
wrpd = warp(a[:,:,:,1], tfms[1]);
@test all_nan_eq(ws[:,:,:,1], wrpd[Base.front(indices(a))...])
wrpd = warp(a[:,:,:,2], tfms[2]);
@test all_nan_eq(ws[:,:,:,2], wrpd[Base.front(indices(a))...])

tfms2 = fill(Translation(1.1,1.2,1.15), n);
ws2 = warpedseries(a, tfms2);
@test size(ws2,1) == size(a,1) - 1
@test size(ws2,2) == size(a,2) - 1
@test size(ws2,3) == size(a,3) - 1

sls = ws2[:,:,:,2:4];
for (n,i) in enumerate(2:4)
    @test all_nan_eq(ws2[:,:,:,i], sls[:,:,:,n])
end
