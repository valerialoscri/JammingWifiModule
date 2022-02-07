/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2010 Network Security Lab, University of Washington, Seattle.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation;
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 * Authors: Sidharth Nabar <snabar@uw.edu>, He Wu <mdzz@u.washington.edu>
 */

#include "mitigate-by-channel-hop.h"
#include "ns3/simulator.h"
#include "ns3/assert.h"
#include "ns3/log.h"
#include "ns3/double.h"
#include "ns3/uinteger.h"
#include "ns3/string.h"
#include <math.h>
#include "ns3/rng-seed-manager.h"
#include "ns3/opengym-module.h"
#include <iostream>
#include <fstream>
#include <string>

#include "ns3/core-module.h"
#include "ns3/applications-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/ipv4-global-routing-helper.h"

NS_LOG_COMPONENT_DEFINE ("MitigateByChannelHop");

namespace ns3 {

NS_OBJECT_ENSURE_REGISTERED (MitigateByChannelHop);

TypeId
MitigateByChannelHop::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::MitigateByChannelHop")
      .SetParent<JammingMitigation> ()
      .AddConstructor<MitigateByChannelHop> ()
      .AddAttribute ("MitigateByChannelHopDetectionMethod",
                     "Jamming detection method to use.",
                     UintegerValue (1), // default to RSS only
                     MakeUintegerAccessor (&MitigateByChannelHop::SetJammingDetectionMethod,
                                           &MitigateByChannelHop::GetJammingDetectionMethod),
                     MakeUintegerChecker<uint32_t> ())
      .AddAttribute ("MitigateByChannelHopDetectionThreshold",
                     "Jamming detection threshold.",
                     DoubleValue (0.5), // default to 0.5
                     MakeDoubleAccessor (&MitigateByChannelHop::SetJammingDetectionThreshold,
                                         &MitigateByChannelHop::GetJammingDetectionThreshold),
                     MakeDoubleChecker<double> ())
      .AddAttribute ("MitigateByChannelHopTxPower",
                     "TX power for channel hop message.",
                     DoubleValue (0.001), // 0.001 W = 0 dBm
                     MakeDoubleAccessor (&MitigateByChannelHop::SetTxPower,
                                         &MitigateByChannelHop::GetTxPower),
                     MakeDoubleChecker<double> ())
      .AddAttribute ("MitigateByChannelHopChannelHopMessage",
                     "Content of channel hop message.",
                     StringValue ("Channel Hop!"),
                     MakeStringAccessor (&MitigateByChannelHop::SetChannelHopMessage,
                                         &MitigateByChannelHop::GetChannelHopMessage),
                     MakeStringChecker ())
      .AddAttribute ("MitigateByChannelHopChannelHopDelay",
                     "Channel hop delay.",
                     TimeValue (Seconds (0.001)),
                     MakeTimeAccessor (&MitigateByChannelHop::SetChannelHopDelay,
                                       &MitigateByChannelHop::GetChannelHopDelay),
                     MakeTimeChecker ())
      .AddAttribute ("MitigateByChannelHopChannelHopSeed",
                     "Seed used in internal RNG.",
                     UintegerValue (12345), // same default defined in rng-stream.h
                     MakeUintegerAccessor (&MitigateByChannelHop::SetRngSeed,
                                           &MitigateByChannelHop::GetRngSeed),
                     MakeUintegerChecker<uint32_t> ())
      .AddAttribute ("MitigateByChannelHopChannelStart",
                     "Starting channel number.",
                     UintegerValue (1),   // first available wifi channel number
                     MakeUintegerAccessor (&MitigateByChannelHop::SetStartChannelNumber,
                                           &MitigateByChannelHop::GetStartChannelNumber),
                     MakeUintegerChecker<uint16_t> ())
      .AddAttribute ("MitigateByChannelHopChannelEnd",
                     "Ending channel number.",
                     UintegerValue (11),  // last available wifi channel number
                     MakeUintegerAccessor (&MitigateByChannelHop::SetEndChannelNumber,
                                           &MitigateByChannelHop::GetEndChannelNumber),
                     MakeUintegerChecker<uint16_t> ())
      .AddAttribute ("MitigateByChannelHopMethod",
                     "Mitigation method to use.",
                     UintegerValue (0), // default to FREQUENCY
                     MakeUintegerAccessor (&MitigateByChannelHop::SetJammingMitigationMethod,
                                           &MitigateByChannelHop::GetJammingMitigationMethod),
                      MakeUintegerChecker<uint32_t> ())
  ;
  return tid;
}
MitigateByChannelHop::MitigateByChannelHop ()
  :  m_rngInitialized (false),
     m_waitingToHop (false),
     m_currentChannel(1),
     m_observation(1),
     m_currentObservation(0),
     m_numberDetection(0)
{
   uint64_t nextStream = RngSeedManager::GetNextStreamIndex ();
   NS_ASSERT(nextStream <= ((1ULL)<<63));
   m_stream = new RngStream (RngSeedManager::GetSeed (),
                                nextStream,
                                RngSeedManager::GetRun ());
}

MitigateByChannelHop::~MitigateByChannelHop ()
{
}

void
MitigateByChannelHop::SetUtility (Ptr<WirelessModuleUtility> utility)
{
  NS_LOG_FUNCTION (this << utility);
  NS_ASSERT (utility != NULL);
  m_utility = utility;
}

void 
MitigateByChannelHop::ChangeChildChannel(uint32_t channel) 
{
  m_utility->SwitchChannel(channel);
}

void
MitigateByChannelHop::SetEnergySource (Ptr<EnergySource> source)
{
  NS_LOG_FUNCTION (this << source);
  NS_ASSERT (source != NULL);
  m_source = source;
}

void
MitigateByChannelHop::SetJammingDetectionMethod (JammingDetectionMethod method)
{
  NS_LOG_FUNCTION (this << method);
  m_jammingDetectionMethod = method;
}

uint32_t
MitigateByChannelHop::GetJammingDetectionMethod (void) const
{
  NS_LOG_FUNCTION (this);
  return m_jammingDetectionMethod;
}

void
MitigateByChannelHop::SetJammingDetectionThreshold (double threshold)
{
  NS_LOG_FUNCTION (this << threshold);
  m_jammingDetectionThreshold = threshold;
}

uint32_t
MitigateByChannelHop::GetJammingMitigationMethod (void) const 
{
   NS_LOG_FUNCTION (this);
   return m_jammingMitigationMethod;
}

void
MitigateByChannelHop::SetJammingMitigationMethod (JammingMitigationMethod method){
    NS_LOG_FUNCTION (this << method);
    m_jammingMitigationMethod = method;
}

double
MitigateByChannelHop::GetJammingDetectionThreshold (void) const
{
  NS_LOG_FUNCTION (this);
  return m_jammingDetectionThreshold;
}

void
MitigateByChannelHop::SetTxPower (double txPower)
{
  NS_LOG_FUNCTION (this << txPower);
  m_txPower = txPower;
}

double
MitigateByChannelHop::GetTxPower (void) const
{
  NS_LOG_FUNCTION (this);
  return m_txPower;
}

void
MitigateByChannelHop::SetChannelHopMessage (std::string message)
{
  NS_LOG_FUNCTION (this);
  m_channelHopMessage = message;
}

std::string
MitigateByChannelHop::GetChannelHopMessage (void) const
{
  NS_LOG_FUNCTION (this);
  return m_channelHopMessage;
}

void
MitigateByChannelHop::SetChannelHopDelay (Time delay)
{
  NS_LOG_FUNCTION (this << delay);
  m_channelHopDelay = delay;
}

Time
MitigateByChannelHop::GetChannelHopDelay (void) const
{
  NS_LOG_FUNCTION (this);
  return m_channelHopDelay;
}

double
MitigateByChannelHop::DegreeOfJamming (int method)
{
  NS_LOG_FUNCTION (this << method);
  NS_LOG_DEBUG ("MitigateByChannelHop:At Node #" << GetId () <<
                ", Calculating degree of jamming!");

  /*
   * Here we use RSS value at the start of packet as reference for calculating
   * RSS ratio. This is effective for reactive jammers where the RSS is low at
   * the beginning of packet, but the average RSS is higher. However this may
   * not be effective for constant jammers, which results a constantly high RSS
   * value and a low RSS ratio.
   */
  double rssRatio = fabs (m_averageRss - m_startRss) / m_startRss;
  switch (method)
    {
    case PDR_ONLY:
      NS_LOG_DEBUG ("MitigateByChannelHop:At Node #" << GetId () <<
                    ", PDR only!");
      NS_LOG_DEBUG(m_utility->GetPdr ());
      return (1 - m_utility->GetPdr ());
    case RSS_ONLY:
      NS_LOG_DEBUG ("MitigateByChannelHop:At Node #" << GetId () <<
                    ", RSS only!");
      return (rssRatio > 1 ? 1 : rssRatio);  // max = 1
    case PDR_AND_RSS:
      NS_LOG_DEBUG ("MitigateByChannelHop:At Node #" << GetId () <<
                    ", PDR & RSS!");
      // return the average of RSS & PDR
      return ((rssRatio + (1 - m_utility->GetPdr ())) / 2);
    default:
      NS_FATAL_ERROR ("MitigateByChannelHop:At Node #" << GetId () <<
                      ", Unknown jamming detection method!");
      break;
    }
  return -1.0;  // error
}

uint32_t
MitigateByChannelHop::GetNumberDetection(void){
  return m_numberDetection;
}

bool
MitigateByChannelHop::IsJammingDetected (int method)
{
  NS_LOG_FUNCTION (this << method);
  NS_LOG_DEBUG ("MitigateByChannelHop:At Node #" << GetId () <<
                ", Deciding if jamming is detected!"  );

  double degreeOfJamming = DegreeOfJamming (method);
  NS_LOG_FUNCTION (degreeOfJamming);
  NS_LOG_FUNCTION (m_jammingDetectionThreshold);
  if (degreeOfJamming > m_jammingDetectionThreshold)
    {
      NS_LOG_DEBUG ("MitigateByChannelHop:At Node #" << GetId () <<
                    ", Jamming is detected! " << m_utility->GetPhyLayerInfo().currentChannel);
      m_currentObservation = m_currentObservation +1;
      m_numberDetection = m_numberDetection +1;
      if(m_currentObservation == m_observation){
        m_currentObservation = 0;
        return true;
      }
      return false;
    }
  else
    {
      NS_LOG_DEBUG ("MitigateByChannelHop:At Node #" << GetId () <<
                    ", Jamming is NOT detected!");
      return false;
    }
}

void
MitigateByChannelHop::SendChannelHopMessage(void)
{
  NS_LOG_FUNCTION (this);
  NS_LOG_DEBUG ("MitigateByChannelHop:At Node #" << GetId () <<
                ", Sending channel hop message >> \n--\n" << m_channelHopMessage <<
                "\n--");

  // build mitigation packet
  Ptr<Packet> packet = Create<Packet> ((uint8_t *)m_channelHopMessage.c_str (),
                                       m_channelHopMessage.size ());

  NS_LOG_DEBUG ("MitigateByChannelHop:At Node #" << GetId () <<
                ", Sending channel hop packet with power = " << m_txPower << " W");

  // send mitigation signal
  double actualPower = m_utility->SendMitigationMessage (packet, m_txPower);
  if (actualPower != 0.0)
    {
      NS_LOG_DEBUG ("MitigateByChannelHop:At Node #" << GetId () <<
                    ", Channel hop packet sent with power = " << actualPower << " W");
    }
  else
    {
      NS_LOG_ERROR ("MitigateByChannelHop:At Node #" << GetId () <<
                    ", Failed to send channel hop packet!");
    }

  m_waitingToHop = true;  // set waiting for channel hop flag
}

void
MitigateByChannelHop::HopChannel (void)
{
  NS_LOG_FUNCTION (this);
 // NS_ASSERT (m_waitingToHop); // make sure we are waiting to hop channel

  // calculate channel number to hop to
  uint16_t channelNumber;

  switch (m_jammingMitigationMethod)
    {
    case FREQUENCY:
       NS_LOG_DEBUG ("MitigateMethod:At Node #" << GetId () << "FREQUENCY");
       channelNumber = ElementaryGenerator();
       break;
    case RANDOM:
      NS_LOG_DEBUG ("MitigateMethod:At Node #" << GetId () << "RANDOM");
       channelNumber =  RandomSequenceGenerator();
      break;
    case SCAN:
       NS_LOG_DEBUG ("MitigateMethod:At Node #" << GetId () << "SCAN");
       channelNumber =  Scan();
       break;
    default:
       channelNumber = m_currentChannel;
      break;
    }

  // schedule hop channel after sending is complete
  m_utility->SwitchChannel(channelNumber);

  NS_LOG_DEBUG ("MitigateByChannelHop:At Node #" << GetId () << ", Hopping from " <<
                m_utility->GetPhyLayerInfo().currentChannel << " >-> " <<
                channelNumber << ", At " << Simulator::Now ().GetSeconds () << "s");

  m_waitingToHop = false; // reset flag after channel hop
  m_currentChannel = channelNumber;

  NS_LOG_DEBUG(channelNumber << "ygs");
}

void
MitigateByChannelHop::SetRngSeed (uint32_t seed)
{
  NS_LOG_FUNCTION (this << seed);
  m_seed = seed;
  //m_stream.SetPackageSeed (seed);
  //m_stream.InitializeStream ();
}

uint32_t
MitigateByChannelHop::GetRngSeed (void) const
{
  NS_LOG_FUNCTION (this);
  return m_seed;
}

void
MitigateByChannelHop::SetStartChannelNumber (uint16_t channelNumber)
{
  NS_LOG_FUNCTION (this << channelNumber);
  m_channelStart = channelNumber;
}

uint16_t
MitigateByChannelHop::GetStartChannelNumber (void) const
{
  NS_LOG_FUNCTION (this);
  return m_channelStart;
}

void
MitigateByChannelHop::SetEndChannelNumber (uint16_t channelNumber)
{
  NS_LOG_FUNCTION (this);
  m_channelEnd = channelNumber;
}

uint16_t
MitigateByChannelHop::GetEndChannelNumber (void) const
{
  NS_LOG_FUNCTION (this);
  return m_channelEnd;
}

/*
 * Private functions start here.
 */

void
MitigateByChannelHop::DoStart (void)
{
  NS_LOG_FUNCTION (this);
  StartMitigation (); // start mitigation at beginning of simulation
}

void
MitigateByChannelHop::DoDispose (void)
{
  NS_LOG_FUNCTION (this);
  NS_LOG_DEBUG ("MitigateByChannelHop: At node #" << GetId () <<
                " Current channel number = " <<
                m_utility->GetPhyLayerInfo ().currentChannel);
  DoStopMitigation ();
}

void
MitigateByChannelHop::DoMitigation (void)
{
  NS_LOG_FUNCTION (this);
  NS_LOG_DEBUG ("MitigateByChannelHop:At Node #" << GetId () <<
                ", Mitigation started!");

  switch (m_jammingMitigationMethod)
    {
    case RL:
       NS_LOG_DEBUG ("MitigateMethod:At Node #" << GetId () << "RL");
       BeginRLMitigation();
       m_mitigation_RL = true;
       break;
    default:
      m_mitigation_RL = false;
      break;
    }

    DoMitigationConstant();

}

void 
MitigateByChannelHop::DoMitigationConstant(void){

  if (IsJammingDetected (m_jammingDetectionMethod) && m_mitigation_RL==false)
    {
      NS_LOG_DEBUG ("MitigateByChannelHop:At Node #" << GetId () <<
                    ", Sending channel hop message at " <<
                    Simulator::Now ().GetSeconds () << "s");
      HopChannel ();
    }

    if(m_mitigation_RL==true){
    // start Rl algorithm and get result for each iteration
    GetObservationsRl();
    HopChannel();
  }
  

  Simulator::Schedule(Seconds(0.01),&MitigateByChannelHop::DoMitigationConstant,this);



}

void
MitigateByChannelHop::DoStopMitigation (void)
{
  NS_LOG_FUNCTION (this);
  NS_LOG_DEBUG ("MitigateByChannelHop:At Node #" << GetId () <<
                ", Mitigation stopped!");
              
  //m_myGymEnv->NotifySimulationEnd();
}

/*void
MitigateByChannelHop::DoStartRxHandler (Ptr<Packet> packet, double startRss)
{
  NS_LOG_FUNCTION (this << packet << startRss);
  m_startRss = startRss;
}
*/

void
MitigateByChannelHop::Synchronisation (Ptr<Packet> packet, double averageRss)
{
if (packet == NULL)
    {
      NS_LOG_DEBUG ("MitigateByChannelHop:At Node #" << GetId () <<
                    ", Failed to receive current packet!");
      return;
    }

if (FindMessage (packet, m_channelHopMessage))
    {

          NS_LOG_DEBUG ("MitigateByChannelHop:At Node #" << GetId () <<
                        ", Received channel hop message at " <<
                        Simulator::Now ().GetSeconds ());
          m_utility->SwitchChannel(m_currentChannel);

}
}
/*void
MitigateByChannelHop::DoEndRxHandler (Ptr<Packet> packet, double averageRss)
{
  NS_LOG_FUNCTION (this << packet << averageRss);
  NS_LOG_DEBUG ("MitigateByChannelHop:At Node #" << GetId () <<
                ", Handling incoming packet!");

  if (!IsMitigationOn () )
    {
      NS_LOG_DEBUG ("MitigateByChannelHop:At Node #" << GetId () <<
                    ", Mitigation is OFF!");
      return;
    }

  m_averageRss = averageRss;

  if(m_mitigation_RL==true){
    // start Rl algorithm and get result for each iteration
    GetObservationsRl();
  }
  else{
  // detect jamming and lauch 
  if (IsJammingDetected (m_jammingDetectionMethod))
    {
      NS_LOG_DEBUG ("MitigateByChannelHop:At Node #" << GetId () <<
                    ", Sending channel hop message at " <<
                    Simulator::Now ().GetSeconds () << "s");
      SendChannelHopMessage ();
    }
  }

  if (packet == NULL)
    {
      NS_LOG_DEBUG ("MitigateByChannelHop:At Node #" << GetId () <<
                    ", Failed to receive current packet!");
      return;
    }

  // check if incoming packet is channel hop message
  if (FindMessage (packet, m_channelHopMessage))
    {
      if (!m_waitingToHop)  // hop only if current node is not waiting to hop
        {
          NS_LOG_DEBUG ("MitigateByChannelHop:At Node #" << GetId () <<
                        ", Received channel hop message at " <<
                        Simulator::Now ().GetSeconds ());
          
          NS_LOG_DEBUG ("MitigateByChannelHop:At Node #" << GetId () <<
                        ", Passing channel hop message!");
          SendChannelHopMessage ();
        }
      else
        {
          NS_LOG_DEBUG ("MitigateByChannelHop:At Node #" << GetId () <<
                        ", Received channel hop message!" <<
                        " But not hopping because current node is waiting to hop!");
        }
    }
}*/

/*void
MitigateByChannelHop::DoEndTxHandler (Ptr<Packet> packet, double txPower)
{
  NS_LOG_FUNCTION (this << packet << txPower);

  if (m_waitingToHop) // check waiting for channel hop flag
    {
      NS_LOG_DEBUG ("MitigateByChannelHop:At Node #" << GetId () <<
                    ", schedule hop at end of TX!");

      NS_LOG_DEBUG ("MitigateByChannelHop:"<<m_channelHopDelay);
      // schedule channel hop after some time
      Simulator::Schedule (m_channelHopDelay, &MitigateByChannelHop::HopChannel,
                           this);
    }
  else
    {
      NS_LOG_DEBUG ("MitigateByChannelHop:At Node #" << GetId () <<
                    ", Not doing anything at end of TX!");
    }
}
*/
bool
MitigateByChannelHop::FindMessage (Ptr<const Packet> packet, std::string target) const
{
  NS_LOG_FUNCTION (this << packet);
  NS_ASSERT (packet != NULL);

 /*if (packet->GetSize () != target.size ()) // check size of packet first
    {
      NS_LOG_DEBUG ("MitigateByChannelHop:At Node #" << GetId () <<
                    ", Size incorrect, not checking data!");
      return false;
    }*/

  // copy data from packet
  uint8_t data[packet->GetSize()];
  packet->CopyData (data, packet->GetSize ());

  // convert data to string
  std::string dataString;
  dataString.assign ((char*)data, packet->GetSize ());

  // show packet content
  NS_LOG_DEBUG ("MitigateByChannelHop:At Node #" << GetId () <<
                ", Packet size = " << packet->GetSize () << " content is:\n---\n" <<
                dataString << "\n---");

  return (dataString.compare (target) == 0);  // check if is channel hop message
}


uint16_t
MitigateByChannelHop::ElementaryGenerator (void)
{ 
  uint16_t channelNumber= m_utility->GetPhyLayerInfo().currentChannel;
  uint16_t nextChannel = channelNumber +1 ;
  if (nextChannel >= 11)
    {
      nextChannel = 1;  // wrap around and start form 1
    }

return nextChannel;
}
uint16_t
MitigateByChannelHop::RandomSequenceGenerator (void)
{
  NS_LOG_FUNCTION (this);
  double nb_aletoire = m_stream->RandU01();
  NS_LOG_FUNCTION (this<<nb_aletoire);
  NS_LOG_FUNCTION (this<<m_channelStart);
  NS_LOG_FUNCTION (this<<m_channelEnd);
  uint32_t nb_channel = (m_channelEnd - m_channelStart) + 1 ;
  NS_LOG_FUNCTION (this<<nb_channel);
  double nb_divisible = 1.0/nb_channel;
  NS_LOG_FUNCTION (this<<nb_divisible);
  double res = nb_aletoire / nb_divisible;
  NS_LOG_FUNCTION (this<<res);
  uint32_t arrondi = floor(res);
  NS_LOG_FUNCTION (this<<arrondi);
  uint16_t channelNumber = arrondi + m_channelStart;
  //uint32_t nb_channel = (m_channelEnd - m_channelStart) + 1 ;
   //uint16_t channelNumber= m_utility->GetPhyLayerInfo().currentChannel +1;
  //uint16_t channelNumber = m_stream->GetInteger (m_channelStart, m_channelEnd);
  //uint16_t channelNumber = m_stream.RandInt (m_channelStart, m_channelEnd);
  //NS_LOG_DEBUG ("MitigateByChannelHop:At Node #" << GetId () << ", RNG returned " <<
                //channelNumber);
  return channelNumber;
}


void
MitigateByChannelHop::BeginRLMitigation(){
  NS_LOG_FUNCTION(this);
        uint32_t openGymPort=5556;
        Ptr<OpenGymInterface> openGymInterface = CreateObject<OpenGymInterface> (openGymPort);
        m_myGymEnv = CreateObject<MyGymEnv>(m_utility->GetPhyLayerInfo().currentChannel,10);
        m_myGymEnv->SetOpenGymInterface(openGymInterface);
}

void 
MitigateByChannelHop::GetObservationsRl(){
     std::ostringstream oss;
     oss.str ("");
     oss << "/NodeList/";
     double pdr =   m_utility->GetPdr ();
     m_myGymEnv->PerformCca(m_myGymEnv,m_currentChannel,pdr);
     uint32_t channel = m_myGymEnv->GetChoiseChannel();
      NS_LOG_FUNCTION ("Ok choise"<<channel );

      if(m_currentChannel != channel){
        m_waitingToHop = true;
        m_currentChannel = channel;
      }
    // NS_LOG_UNCOND ("MyGetObservation: " << m_myGymEnv->GetObservation());
    //m_myGymEnv->NotifySimulationEnd();
}


uint16_t 
MitigateByChannelHop::Scan(){
double rss = m_utility->GetRss();
NS_LOG_DEBUG("RSS: " << rss );
return 1;
}



} // namespace ns3{

