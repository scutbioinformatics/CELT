function ret = depend( data )
n = 20;
data = data';
d = size(data,2);
ret = zeros(d,d);
for i=1:d
    for j=i+1:d
        ret(i, j) = calmi(data(:,i),data(:,j),n);
    end
end
ret = ret + ret';
for i=1:d
    ret(i, i) = sum(ret(:, i));
end