#ifndef GYM_ACTOR_CRITIC_H
#define GYM_ACTOR_CRITIC_H


#include "ns3/opengym-module.h"
#include "ns3/stats-module.h"



namespace ns3 {
    class Node;
    class Packet;

    class GymActorCritic: public OpenGymEnv

    {
        public: 
            static TypeId GetTypeId(void);
            GymActorCritic();
            virtual ~GymActorCritic();
            GymActorCritic(uint32_t currentChannel, uint32_t channelNum);
            virtual void DoDispose (void);

            
            // Accessors

            uint32_t GetCurrentChannel();
            double GetTxPower();
            void SetCurrentChannel(uint32_t current_channel);
            // CallBack Functions OpenGymNs3

            Ptr<OpenGymSpace> GetObservationSpace();
            Ptr<OpenGymSpace> GetActionSpace();
            Ptr<OpenGymDataContainer> GetObservation();
            bool GetGameOver();
            float GetReward();
            std::string GetExtraInfo();
            bool ExecuteActions(Ptr<OpenGymDataContainer> action);


            //Other Function

            void CollectChannelOccupation(uint16_t chanId, uint32_t occupied);
            bool CheckIfReady();
            void  ClearObs();
            void UpdateActorCriticWithRss(Ptr<GymActorCritic> entity, uint16_t channelId, double avgPower);
            void UpdateActorCriticWithPacket(Ptr<GymActorCritic> entity, uint16_t channelId, uint32_t numberPacket);
    


        private:
            uint16_t m_currentNode;
            uint16_t m_rxPkTNum;
            uint32_t m_channelNum;
            std::vector<uint16_t> m_channelOccupation = {0,0,0,0,0,0,0,0,0,0,0,0};
            uint32_t m_currentChannel;
            uint32_t m_collisionTh;
            std::vector<uint32_t> m_collision = {0,0,0,0,0,0,0,0,0,0,0,0};
        
    };
}

#endif