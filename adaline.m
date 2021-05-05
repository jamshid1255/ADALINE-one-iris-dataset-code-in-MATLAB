clear all
close all

%load iris_data
load iris_data.mat

X=[x(1:100,1) x(1:100,2)];
for i=1:100
    if y(i) == 0
        Y(i,1)=1;
    else
        Y(i,1)=-1;
    end
end

figure(1)
hold on
plot(X(1:50,1),X(1:50,2),'ro')
plot(X(51:100,1),X(51:100,2),'bx')

xlabel('Sepal length')
ylabel('Petal length')

%Perceptron
n_iter=2000;
eta=0.0001;

w=[0.03;0.03;0.03];
sum_square_erro = zeros(n_iter,1);
for i=1:n_iter
    error_sum=zeros(3,1);
    %prediction
    for j=1:length(X)
    Yhat(j,1)=step_f(w'*[1;X(j,:)']);
            
    error_sum = error_sum + (Y(j,1)-Yhat(j,1))*[1;X(j,:)'];
    sum_square_erro(i,1) = sum_square_erro(i,1) + (1/200)*(Y(j,1)-Yhat(j,1))^2;
    end
    
    %update
    w=w+eta*(Y(j,1)-Yhat(j,1))*[1;X(j,:)'];

end


x1=4:0.1:7;
for i=1:length(x1)
    x2(i)=-w(1)/w(3)-w(2)/w(3)*x1(i);
end

figure(1)
plot(x1,x2)

figure(2)
plot(sum_square_erro)

a = [5.1,4.9,4.7,6];
b = [3.5,3,3.2,3.4];
length(a)
for i=1:length(a)
    if step_f(w(2)*a(i) + w(3)*b(i) + w(1)) == 1
        res = 'setosa'
    else
        res = 'versicolor'
    end
end










