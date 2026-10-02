from .treatment.chlorine_ct import compliant
from .treatment.residual_anomaly import ResidualAnomalyService, ReleaseSpec
from .network.isolation import isolated
from .meters.balance import balance
from .leak.localize import rank
from .workflow_service import WorkflowService
class AquaFlowService:
    def __init__(self):
        self.workflows=WorkflowService()
        self.residual_anomalies=ResidualAnomalyService()
    def health(self):return {"service":"aquaflow","status":"ok","runtime_dependencies":0}
    def ct(self,rows,required):return compliant(rows,required)
    def isolate(self,nodes,pipes,closed,sources):return isolated(nodes,pipes,closed,sources)
    def water_balance(self,ins,outs,known_loss):return balance(ins,outs,known_loss)
    def leak_rank(self,rows):return rank(rows)
    def run_workflow(self,name,payload,received_at=None):return self.workflows.run(name,payload,received_at)
    # residual-chlorine anomaly correlation casebook
    def ingest_readings(self,readings):return self.residual_anomalies.ingest(readings)
    def correlate_evidence(self,chain_id,start,end,**kw):return self.residual_anomalies.correlate(chain_id,start,end,**kw).to_dict()
    def open_residual_case(self,chain_id,start,end,**kw):return self.residual_anomalies.open_if_unprovable(chain_id,start,end,**kw)
    def residual_case(self,case_id):return self.residual_anomalies.case_view(case_id)
    def decide_residual_case(self,case_id,action,actor,reason,**kw):return self.residual_anomalies.decide(case_id,action,actor,reason,**kw)
    def residual_cases(self,chain_id=None,status=None):return self.residual_anomalies.cases(chain_id,status)
