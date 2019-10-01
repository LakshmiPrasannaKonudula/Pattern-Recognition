FULLY_TRAINED_DATA=[];
MAX_ITERATION=1000;
Q = readtable('cars.csv');
A=table2array(Q);
[m,n] = size(A) ;
P = 0.90 ;
%Dividing the data set into 90% testing and 10% training
idx = randperm(m)  ;
Training = A(idx(1:round(P*m)),:) ; 
Testing = A(idx(round(P*m)+1:end),:) ;


for ITERATION = 1:MAX_ITERATION
    
RandomRowNumber=randi(size(Training,1)); %Selecting a random vector from matrix to select conditional variables
randomRowVector = Training(RandomRowNumber,:); 
rv_buyPrice=randomRowVector(1,1);
rv_maint=randomRowVector(1,2);
rv_doors=randomRowVector(1,3);
rv_persons=randomRowVector(1,4);
rv_lug_boot=randomRowVector(1,5);
rv_safety=randomRowVector(1,6);
rv_acceptable=randomRowVector(1,7);

fprintf('Input Vector Buy price is: %s\n', rv_buyPrice{:});
fprintf('Input Vector maint is: %s\n', rv_maint{:});
fprintf('Input Vector doors is: %s\n', rv_doors{:});
fprintf('Input Vector persons is: %s\n', rv_persons{:});
fprintf('Input Vector lug_boot is: %s\n', rv_lug_boot{:});
fprintf('Input Vector safety is: %s\n', rv_safety{:});
fprintf('Input Vector acceptablity is: %s\n', rv_acceptable{:});
%Implementing counting routines to calculate individual probabilities
countBuyingPrice(rv_buyPrice,Training);
countMaint(rv_maint,Training);
countSafety(rv_safety,Training);
countDoors(rv_doors,Training);
countPersons(rv_persons,Training);
countLugBoot(rv_lug_boot,Training);
countAcceptable(rv_acceptable,Training);

m_y = size(Training,1) ;
p_y_sum=sum(ismember(Training(:,7),rv_acceptable));
prob_y=p_y_sum/m_y;
prob_buy_price=find_indv_prob(Training,1,rv_buyPrice{1},rv_acceptable{1});%Probabilities of buying price with condition 'acc'
prob_maint=find_indv_prob(Training,2,rv_maint{1},rv_acceptable{1});%Probabilities of maintainence with condition 'acc'
prob_safety=find_indv_prob(Training,6,rv_safety{1},rv_acceptable{1});%Probabilities of safety with condition 'acc'

prob_ALL=find_all_prob(Training,rv_buyPrice{1},rv_maint{1},rv_safety{1});%Calculating total probability with all conditions
[bp_prob_acc,bp_prob_unacc,bp_prob_good,bp_prob_vgood]=prob_all_cases(Training,1,rv_buyPrice{1});
[mt_prob_acc,mt_prob_unacc,mt_prob_good,mt_prob_vgood]=prob_all_cases(Training,2,rv_maint{1});
[sf_prob_acc,sf_prob_unacc,sf_prob_good,sf_prob_vgood]=prob_all_cases(Training,6,rv_safety{1});
disp('*************************************');
    
PROB_COMB_ACC=bp_prob_acc * mt_prob_acc * sf_prob_acc;
disp(PROB_COMB_ACC);%%Calculating posterior class probability	for	each	value	of	the	favorable variable
PROB_COMB_UNACC=bp_prob_unacc * mt_prob_unacc * sf_prob_unacc;
disp(PROB_COMB_UNACC);
PROB_COMB_GOOD=bp_prob_good * mt_prob_good * sf_prob_good;
disp(PROB_COMB_GOOD);
PROB_COMB_VGOOD=bp_prob_vgood * mt_prob_vgood * sf_prob_vgood;
disp(PROB_COMB_VGOOD);
%Calculating Bayes probabilty and storing train data
TRAINED_DATA=[rv_buyPrice rv_maint rv_safety PROB_COMB_ACC PROB_COMB_UNACC PROB_COMB_GOOD PROB_COMB_VGOOD];
DUP_FULLY_TRAINED_DATA=[FULLY_TRAINED_DATA;TRAINED_DATA];
[~,idx]=unique(strcat(DUP_FULLY_TRAINED_DATA(:,1),DUP_FULLY_TRAINED_DATA(:,2),DUP_FULLY_TRAINED_DATA(:,3),DUP_FULLY_TRAINED_DATA(:,4),DUP_FULLY_TRAINED_DATA(:,5),DUP_FULLY_TRAINED_DATA(:,6),DUP_FULLY_TRAINED_DATA(:,7)) );
FULLY_TRAINED_DATA=DUP_FULLY_TRAINED_DATA(idx,:)
end
%Testing the data using previous probabilties and if the value is unknown
%it is calculated using existing probabilities with MAP decision rule

TEST_SIZE=size(Testing,1);
TEST_RESULTS=[];
MATCH_TEST=0;
for ITERATION_TEST = 1:TEST_SIZE

TEST_VECTOR=Testing(ITERATION_TEST,:);
    disp(TEST_VECTOR);
    test_class=TEST_VECTOR(1,7);
    OUTPUT_TEST=FULLY_TRAINED_DATA(strcmp(FULLY_TRAINED_DATA(:,1),TEST_VECTOR(:,1)) & strcmp(FULLY_TRAINED_DATA(:,2),TEST_VECTOR(:,2)) & strcmp(FULLY_TRAINED_DATA(:,3),TEST_VECTOR(:,6)),:);
    disp('###THIS IS OUTPUT_TEST###');
    if isempty(OUTPUT_TEST)
        TESTING_RESULT='NOT DETERMINED'
    else
        disp(OUTPUT_TEST);
    TEST_ACC=OUTPUT_TEST{4};
    TEST_UNACC=OUTPUT_TEST{5};
    TEST_GOOD=OUTPUT_TEST{6};
    TEST_VGOOD=OUTPUT_TEST{7};
    disp('########################');
    fprintf(' Probability of ACC is: %.5f\n', TEST_ACC);
    fprintf(' Probability of UNACC is: %.5f\n', TEST_UNACC);
    fprintf(' Probability of GOOD is: %.5f\n', TEST_GOOD);
    fprintf(' Probability of VGOOD is: %.5f\n', TEST_VGOOD);
    P=[TEST_ACC,TEST_UNACC,TEST_GOOD,TEST_VGOOD];
    MAX_VALUE=max(P(:));
    disp('########################');
    fprintf('Maximum Probability is: %.5f\n', MAX_VALUE);
    if(MAX_VALUE==TEST_ACC) 
        TESTING_RESULT='acc'
    else if(MAX_VALUE==TEST_UNACC) 
            TESTING_RESULT='unacc'
        else if(MAX_VALUE==TEST_GOOD) 
                TESTING_RESULT='good'
            else if(MAX_VALUE==TEST_VGOOD) 
                    TESTING_RESULT='vgood'                    
                end
            end
        end
    end
    end
    if(string(TESTING_RESULT)==string(test_class{1}))
        MATCH_TEST=MATCH_TEST+1;
    end
    
    INTER_TEST=[TEST_VECTOR TESTING_RESULT];
    TEST_RESULTS=[TEST_RESULTS; INTER_TEST];       
end
ACCURACY=(MATCH_TEST/TEST_SIZE)*100;
fprintf('Accuracy is: %.5f\n', ACCURACY);

function prob_each = find_indv_prob(dataArray,field_no,field_value,value)
     field_and_y_match=dataArray(strcmp(dataArray(:,field_no),field_value) & strcmp(dataArray(:,7),value),:);
     y_match=dataArray(strcmp(dataArray(:,7),value),:);
     Y_size_m=size(y_match,1);
     F_Y_size_m=size(field_and_y_match,1);
     prob_each=F_Y_size_m/Y_size_m;
end

function prob_all = find_all_prob(dataArray,field_1_value,field_2_value,field_3_value)
     ALL_Match = dataArray(strcmp(dataArray(:,1),field_1_value) & strcmp(dataArray(:,2),field_2_value) & strcmp(dataArray(:,6),field_3_value),:);
     X_size_m=size(dataArray,1);
     Y_size_m=size(ALL_Match,1);
     prob_all=Y_size_m/X_size_m;
end
function [P1_FIELD_ACC,P1_FIELD_UNACC,P1_FIELD_GOOD,P1_FIELD_VGOOD]=prob_all_cases(dataArray,field_position,field_value)
    ACC_MATCH=dataArray(strcmp(dataArray(:,field_position),field_value) & strcmp(dataArray(:,7),'acc'),:);
    UNACC_MATCH=dataArray(strcmp(dataArray(:,field_position),field_value) & strcmp(dataArray(:,7),'unacc'),:);
    GOOD_MATCH=dataArray(strcmp(dataArray(:,field_position),field_value) & strcmp(dataArray(:,7),'good'),:);
    VGOOD_MATCH=dataArray(strcmp(dataArray(:,field_position),field_value) & strcmp(dataArray(:,7),'vgood'),:);
    ACC_FIELD_COUNT=size(ACC_MATCH,1);
    UNACC_FIELD_COUNT=size(UNACC_MATCH,1);
    GOOD_FIELD_COUNT=size(GOOD_MATCH,1);
    VGOOD_FIELD_COUNT=size(VGOOD_MATCH,1);
    
    FIELD_MATCH_COUNT=GET_MATCH_COUNT(dataArray,field_position,field_value)
    TOTAL_COUNT=size(dataArray,1);
    
    ACC_COUNT=GET_MATCH_COUNT(dataArray,7,'acc');
    UNACC_COUNT=GET_MATCH_COUNT(dataArray,7,'unacc');
    GOOD_COUNT=GET_MATCH_COUNT(dataArray,7,'good');
    VGOOD_COUNT=GET_MATCH_COUNT(dataArray,7,'vgood');
    prob_acc=ACC_COUNT/TOTAL_COUNT;
    prob_unacc=UNACC_COUNT/TOTAL_COUNT;
    prob_good=GOOD_COUNT/TOTAL_COUNT;
    prob_vgood=VGOOD_COUNT/TOTAL_COUNT;

    
    prob_field_acc=ACC_FIELD_COUNT/FIELD_MATCH_COUNT;
    prob_field_unacc=UNACC_FIELD_COUNT/FIELD_MATCH_COUNT;
    prob_field_good=GOOD_FIELD_COUNT/FIELD_MATCH_COUNT;
    prob_field_vgood=VGOOD_FIELD_COUNT/FIELD_MATCH_COUNT;
    prob_field=FIELD_MATCH_COUNT/TOTAL_COUNT;
    
    P1_FIELD_ACC=(prob_field_acc*prob_field)/prob_acc;
    P1_FIELD_UNACC=(prob_field_unacc*prob_field)/prob_unacc;
    P1_FIELD_GOOD=(prob_field_good*prob_field)/prob_good;
    P1_FIELD_VGOOD=(prob_field_vgood*prob_field)/prob_vgood;
    
    

    
end
function MATCH_COUNT=GET_MATCH_COUNT(dataArray,field_position,field_value)
    MATCH_DATA=dataArray(strcmp(dataArray(:,field_position),field_value),:);
    MATCH_COUNT=size(MATCH_DATA,1);
end
function rv_buyprice_count= countBuyingPrice(value, dataArray)
    rv_buyprice_count=sum(ismember(dataArray(:,1),value))
    c=categorical(dataArray(:,1));
    disp('Unique Buying Price values in data array:');
    summary(c);
end

function rv_maint_count= countMaint(value, dataArray)
    rv_maint_count=sum(ismember(dataArray(:,2),value))
    c=categorical(dataArray(:,2));
    disp('Unique Maint values in data array:');
    summary(c);
end

function rv_safety_count= countSafety(value, dataArray)
    rv_safety_count=sum(ismember(dataArray(:,6),value))
    c=categorical(dataArray(:,6));
    disp('Unique safety values in data array:');
    summary(c);
end
function rv_doors_count= countDoors(value, dataArray)
    rv_doors_count=sum(ismember(dataArray(:,3),value))
    c=categorical(dataArray(:,3));
    disp('Unique doors values in data array:');
    summary(c);
end
function rv_persons_count= countPersons(value, dataArray)
    rv_persons_count=sum(ismember(dataArray(:,4),value))
    c=categorical(dataArray(:,4));
    disp('Unique persons values in data array:');
    summary(c);
end
function rv_lug_boot_count= countLugBoot(value, dataArray)
    rv_lug_boot_count=sum(ismember(dataArray(:,5),value))
    c=categorical(dataArray(:,5));
    disp('Unique lug_boot values in data array:');
    summary(c);
end
function rv_acceptable_count= countAcceptable(value, dataArray)
    rv_acceptable_count=sum(ismember(dataArray(:,7),value))
    c=categorical(dataArray(:,7));
    disp('Unique Acceptable values in data array:');
    summary(c);
end