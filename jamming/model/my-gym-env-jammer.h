#ifndef MY_GYM_ENV_JAMMER_H
#define MY_GYM_ENV_JAMMER_H

#include "ns3/opengym-module.h"
#include "ns3/stats-module.h"

namespace ns3 {
    class Node;
    class Packet;

    class MyGymEnvJammer : public OpenGymEnv
    {
        public:
            MyGymEnvJammer();
            MyGymEnvJammer(uint32_t currentChannel, uint32_t channelNum);
            virtual ~MyGymEnvJammer ();
            static TypeId GetTypeId (void);
            virtual void DoDispose ();

            Ptr<OpenGymSpace> GetActionSpace();
            Ptr<OpenGymSpace> GetObservationSpace();
            bool GetGameOver();
            Ptr<OpenGymDataContainer> GetObservation();
            float GetReward();
            std::string GetExtraInfo();
            bool ExecuteActions(Ptr<OpenGymDataContainer> action);

            static void PerformCca(Ptr<MyGymEnvJammer> entity,uint16_t channelId,  double avgPowerSpectralDensity);
            static void PerformJamming(Ptr<MyGymEnvJammer> entity,uint16_t channelId,  double avgPowerSpectralDensity);
            void CollectChannelOccupation(uint16_t chanId, uint32_t occupied);
            bool CheckIfReady();
            void ClearObs();
            uint32_t GetChoiseChannel(void);

        private:
            void ScheduleNextStateRead();
  

            Time m_interval = Seconds(0.1);
            uint16_t m_currentNode;
            uint64_t m_rxPktNum;
            uint32_t m_channelNum;
            std::vector<uint16_t> m_channelOccupation ;
            uint32_t m_currentChannel;
            uint32_t m_collisionTh;
            std::vector<uint32_t> m_collisions = {0,0,0,0,0,0,0,0,0,0,0,0};
            

    };
}
#endif