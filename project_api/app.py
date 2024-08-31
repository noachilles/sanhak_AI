from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import numpy as np
import xgboost
import pickle

app = Flask(__name__)
CORS(app)


model = pickle.load(open("xgboost_model.pkl", "rb"))

def fill_columns(n, data, name):
    data = format(data, 'b').zfill(n)
    print(data)
    for i in range(len(data)):
        name.append(int(data[i]))

@app.route('/predict', methods=["POST"])
def home():
    try:
        insured_ocp_lst = []
        insured_sex_lst = []
        collision_type_lst = []
        incident_state_lst = []
        authorities_contacted_lst = []
        property_damage_lst = []

        data = request.get_json(force=True)
        
        insured_ocp = int(data['insured_occupation'])
        insured_sex = int(data['insured_sex'])
        incident_severity = int(data['incident_severity'])
        collision_type = int(data['collision_type'])
        incident_state = int(data['incident_state'])
        number_of_vehicles_involved = int(data['number_of_vehicles_involved'])
        witnesses = int(data['witnesses'])
        authorities_contacted = int(data['authorities_contacted'])
        property_damage = int(data['property_damage'])
        policy_deductable = int(data['policy_deductable'])
        total_claim_amount = int(data['total_claim_amount'])
        property_claim = int(data['property_claim'])
        vehicle_claim = int(data['vehicle_claim'])
        
        fill_columns(4, insured_ocp, insured_ocp_lst)
        fill_columns(2, insured_sex, insured_sex_lst)
        fill_columns(3, collision_type, collision_type_lst)
        fill_columns(3, incident_state, incident_state_lst)
        fill_columns(3, authorities_contacted, authorities_contacted_lst)
        fill_columns(2, property_damage, property_damage_lst)
        

        arr = np.array([[policy_deductable, 
                        insured_sex_lst[0], insured_sex_lst[1], 
                        insured_ocp_lst[0], insured_ocp_lst[1], insured_ocp_lst[2], insured_ocp_lst[3], 
                        collision_type_lst[0], collision_type_lst[1], collision_type_lst[2], 
                        incident_severity, 
                        authorities_contacted_lst[0], authorities_contacted_lst[1], authorities_contacted_lst[2], 
                        incident_state_lst[0], incident_state_lst[1], incident_state_lst[2], 
                        number_of_vehicles_involved, property_damage_lst[0], property_damage_lst[1], 
                        witnesses, total_claim_amount, property_claim, vehicle_claim]])
        pred = model.predict_proba(arr)
        pred = pred[:,1] * 100
        pred = np.round(pred).astype(int)
        print(f"Prediction: {int(pred[0])}")
        # return render_template("after.html", pred=pred[0])
        return jsonify({'prediction': int(pred[0])})
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)})
    
    

        

if __name__ == "__main__":
    app.run(debug=True, port=5000)
    

    # (bin)/(ord) 표시된 변수는 입력 받을 때 category로 선택
    # 변수 내 value 값 옆에 표시된 숫자를 request value로 가짐
    
    # 변수 구분
    # insured_occupation - 보험계약자의 직업 (bin) 1~14
        # machine-op-inspct 1
        # prof-specialty 2
        # tech-support 3
        # sales 4
        # exec-managerial 5
        # craft-repair 6
        # transport-moving 7
        # other-service 8
        # priv-house-serv 9
        # armed-forces 10
        # adm-clerical 11
        # protective-serv 12
        # handlers-cleaners 13
        # farming-fishing 14

        
    # insured_sex - 보험계약자 성별 (bin) 1~2
        # Female 1
        # Male 2
    # -----------------------------------
    # incident_severity - 사건 심각성 (ord) 1~4
        # Trival Damage 1
        # Minor Damage 2
        # Major Damage 3
        # Total Loss 4
        
    # collision_type - 발생 충돌 유형 (bin) 1~3
        # Rear Collision 1
        # Side Collision 2
        # Front Collision 3
        # UNSURE 4
    # incident_state - 사건 발생 주(state) (bin) 1~7
        # NY 1
        # SC 2
        # WV 3
        # VA 4
        # NC 5
        # PA 6
        # OH 7
    # number_of_vehicles_involved - 사건 관련된 차량 수 
    # witnesses - 목격자 수 
    # authorities_contacted - 연락된 당국 (bin) 1~4
        # Police 1
        # Fire 2
        # Other 3
        # Ambulance 4
    # -----------------------------------
    # property_damage - 재산 피해 (bin) 1~2
        # NO 1
        # YES 2
    # policy_deductable - 배상 전 보험 계약자가 지불해야 하는 금액
    # total_claim_amount - 고객이 청구한 총 금액
    # property_claim - 재산 손해 청구 금액
    # vehicle_claim - 차량 손해 청구 금액
    