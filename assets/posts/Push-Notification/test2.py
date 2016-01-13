from gcm import GCM

gcm = GCM('AIzaSyB2UGIEf-3dVZvtT0CSXzTmrLhGCsMd1XE')
data = {'message': 'hihihi'}

# Downstream message using JSON request
reg_ids = ['eQRPiEhsH4w:APA91bHY1BpRKeXMCgS1Vr1CphgIbMvuVezjSIY1WwJf9l2AsFvqcUzV55C9drEVg1eSvBHCA8zwHkpxlP2zG8YG5umpIrFunkclTcJNp6Euzv49iIttwRbBmAAwUNICN9HRSgVazXoy']
response = gcm.json_request(registration_ids=reg_ids, data=data)