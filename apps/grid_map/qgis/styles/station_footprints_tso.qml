<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.34.0-Prizren" styleCategories="Symbology|Labeling">
  <renderer-v2 type="RuleRenderer" symbollevels="0" enableorderby="0" forceraster="0" referencescale="-1">
    <rules key="{root}">
      <rule key="{r0}" label="50Hertz, footprint (OpenStreetMap)" symbol="0" filter="(&quot;tso&quot; = '50Hertz') AND &quot;source&quot; &lt;&gt; 'box'" scalemaxdenom="70000" scalemindenom="1" />
      <rule key="{r1}" label="50Hertz, box (no footprint)" symbol="1" filter="(&quot;tso&quot; = '50Hertz') AND &quot;source&quot; = 'box'" scalemaxdenom="70000" scalemindenom="1" />
      <rule key="{r2}" label="Amprion, footprint (OpenStreetMap)" symbol="2" filter="(&quot;tso&quot; = 'Amprion') AND &quot;source&quot; &lt;&gt; 'box'" scalemaxdenom="70000" scalemindenom="1" />
      <rule key="{r3}" label="Amprion, box (no footprint)" symbol="3" filter="(&quot;tso&quot; = 'Amprion') AND &quot;source&quot; = 'box'" scalemaxdenom="70000" scalemindenom="1" />
      <rule key="{r4}" label="TenneT, footprint (OpenStreetMap)" symbol="4" filter="(&quot;tso&quot; = 'TenneT') AND &quot;source&quot; &lt;&gt; 'box'" scalemaxdenom="70000" scalemindenom="1" />
      <rule key="{r5}" label="TenneT, box (no footprint)" symbol="5" filter="(&quot;tso&quot; = 'TenneT') AND &quot;source&quot; = 'box'" scalemaxdenom="70000" scalemindenom="1" />
      <rule key="{r6}" label="TransnetBW, footprint (OpenStreetMap)" symbol="6" filter="(&quot;tso&quot; = 'TransnetBW') AND &quot;source&quot; &lt;&gt; 'box'" scalemaxdenom="70000" scalemindenom="1" />
      <rule key="{r7}" label="TransnetBW, box (no footprint)" symbol="7" filter="(&quot;tso&quot; = 'TransnetBW') AND &quot;source&quot; = 'box'" scalemaxdenom="70000" scalemindenom="1" />
      <rule key="{r8}" label="Other, footprint (OpenStreetMap)" symbol="8" filter="(&quot;tso&quot; NOT IN ('50Hertz', 'Amprion', 'TenneT', 'TransnetBW')) AND &quot;source&quot; &lt;&gt; 'box'" scalemaxdenom="70000" scalemindenom="1" />
      <rule key="{r9}" label="Other, box (no footprint)" symbol="9" filter="(&quot;tso&quot; NOT IN ('50Hertz', 'Amprion', 'TenneT', 'TransnetBW')) AND &quot;source&quot; = 'box'" scalemaxdenom="70000" scalemindenom="1" />
    </rules>
    <symbols>
      <symbol type="fill" name="0" alpha="1" clip_to_extent="1" force_rhr="0" frame_rate="10" is_animated="0">
        <layer class="SimpleFill" enabled="1" locked="0" pass="0">
          <Option type="Map">
            <Option name="color" type="QString" value="42,120,214,38" />
            <Option name="style" type="QString" value="solid" />
            <Option name="outline_color" type="QString" value="42,120,214,255" />
            <Option name="outline_style" type="QString" value="solid" />
            <Option name="outline_width" type="QString" value="0.5" />
            <Option name="outline_width_unit" type="QString" value="MM" />
            <Option name="joinstyle" type="QString" value="round" />
          </Option>
          <data_defined_properties>
            <Option type="Map">
              <Option name="name" type="QString" value="" />
              <Option name="properties" type="Map" />
              <Option name="type" type="QString" value="collection" />
            </Option>
          </data_defined_properties>
        </layer>
      </symbol>
      <symbol type="fill" name="1" alpha="1" clip_to_extent="1" force_rhr="0" frame_rate="10" is_animated="0">
        <layer class="SimpleFill" enabled="1" locked="0" pass="0">
          <Option type="Map">
            <Option name="color" type="QString" value="42,120,214,38" />
            <Option name="style" type="QString" value="solid" />
            <Option name="outline_color" type="QString" value="42,120,214,255" />
            <Option name="outline_style" type="QString" value="dash" />
            <Option name="outline_width" type="QString" value="0.5" />
            <Option name="outline_width_unit" type="QString" value="MM" />
            <Option name="joinstyle" type="QString" value="round" />
          </Option>
          <data_defined_properties>
            <Option type="Map">
              <Option name="name" type="QString" value="" />
              <Option name="properties" type="Map" />
              <Option name="type" type="QString" value="collection" />
            </Option>
          </data_defined_properties>
        </layer>
      </symbol>
      <symbol type="fill" name="2" alpha="1" clip_to_extent="1" force_rhr="0" frame_rate="10" is_animated="0">
        <layer class="SimpleFill" enabled="1" locked="0" pass="0">
          <Option type="Map">
            <Option name="color" type="QString" value="235,104,52,38" />
            <Option name="style" type="QString" value="solid" />
            <Option name="outline_color" type="QString" value="235,104,52,255" />
            <Option name="outline_style" type="QString" value="solid" />
            <Option name="outline_width" type="QString" value="0.5" />
            <Option name="outline_width_unit" type="QString" value="MM" />
            <Option name="joinstyle" type="QString" value="round" />
          </Option>
          <data_defined_properties>
            <Option type="Map">
              <Option name="name" type="QString" value="" />
              <Option name="properties" type="Map" />
              <Option name="type" type="QString" value="collection" />
            </Option>
          </data_defined_properties>
        </layer>
      </symbol>
      <symbol type="fill" name="3" alpha="1" clip_to_extent="1" force_rhr="0" frame_rate="10" is_animated="0">
        <layer class="SimpleFill" enabled="1" locked="0" pass="0">
          <Option type="Map">
            <Option name="color" type="QString" value="235,104,52,38" />
            <Option name="style" type="QString" value="solid" />
            <Option name="outline_color" type="QString" value="235,104,52,255" />
            <Option name="outline_style" type="QString" value="dash" />
            <Option name="outline_width" type="QString" value="0.5" />
            <Option name="outline_width_unit" type="QString" value="MM" />
            <Option name="joinstyle" type="QString" value="round" />
          </Option>
          <data_defined_properties>
            <Option type="Map">
              <Option name="name" type="QString" value="" />
              <Option name="properties" type="Map" />
              <Option name="type" type="QString" value="collection" />
            </Option>
          </data_defined_properties>
        </layer>
      </symbol>
      <symbol type="fill" name="4" alpha="1" clip_to_extent="1" force_rhr="0" frame_rate="10" is_animated="0">
        <layer class="SimpleFill" enabled="1" locked="0" pass="0">
          <Option type="Map">
            <Option name="color" type="QString" value="27,175,122,38" />
            <Option name="style" type="QString" value="solid" />
            <Option name="outline_color" type="QString" value="27,175,122,255" />
            <Option name="outline_style" type="QString" value="solid" />
            <Option name="outline_width" type="QString" value="0.5" />
            <Option name="outline_width_unit" type="QString" value="MM" />
            <Option name="joinstyle" type="QString" value="round" />
          </Option>
          <data_defined_properties>
            <Option type="Map">
              <Option name="name" type="QString" value="" />
              <Option name="properties" type="Map" />
              <Option name="type" type="QString" value="collection" />
            </Option>
          </data_defined_properties>
        </layer>
      </symbol>
      <symbol type="fill" name="5" alpha="1" clip_to_extent="1" force_rhr="0" frame_rate="10" is_animated="0">
        <layer class="SimpleFill" enabled="1" locked="0" pass="0">
          <Option type="Map">
            <Option name="color" type="QString" value="27,175,122,38" />
            <Option name="style" type="QString" value="solid" />
            <Option name="outline_color" type="QString" value="27,175,122,255" />
            <Option name="outline_style" type="QString" value="dash" />
            <Option name="outline_width" type="QString" value="0.5" />
            <Option name="outline_width_unit" type="QString" value="MM" />
            <Option name="joinstyle" type="QString" value="round" />
          </Option>
          <data_defined_properties>
            <Option type="Map">
              <Option name="name" type="QString" value="" />
              <Option name="properties" type="Map" />
              <Option name="type" type="QString" value="collection" />
            </Option>
          </data_defined_properties>
        </layer>
      </symbol>
      <symbol type="fill" name="6" alpha="1" clip_to_extent="1" force_rhr="0" frame_rate="10" is_animated="0">
        <layer class="SimpleFill" enabled="1" locked="0" pass="0">
          <Option type="Map">
            <Option name="color" type="QString" value="237,161,0,38" />
            <Option name="style" type="QString" value="solid" />
            <Option name="outline_color" type="QString" value="237,161,0,255" />
            <Option name="outline_style" type="QString" value="solid" />
            <Option name="outline_width" type="QString" value="0.5" />
            <Option name="outline_width_unit" type="QString" value="MM" />
            <Option name="joinstyle" type="QString" value="round" />
          </Option>
          <data_defined_properties>
            <Option type="Map">
              <Option name="name" type="QString" value="" />
              <Option name="properties" type="Map" />
              <Option name="type" type="QString" value="collection" />
            </Option>
          </data_defined_properties>
        </layer>
      </symbol>
      <symbol type="fill" name="7" alpha="1" clip_to_extent="1" force_rhr="0" frame_rate="10" is_animated="0">
        <layer class="SimpleFill" enabled="1" locked="0" pass="0">
          <Option type="Map">
            <Option name="color" type="QString" value="237,161,0,38" />
            <Option name="style" type="QString" value="solid" />
            <Option name="outline_color" type="QString" value="237,161,0,255" />
            <Option name="outline_style" type="QString" value="dash" />
            <Option name="outline_width" type="QString" value="0.5" />
            <Option name="outline_width_unit" type="QString" value="MM" />
            <Option name="joinstyle" type="QString" value="round" />
          </Option>
          <data_defined_properties>
            <Option type="Map">
              <Option name="name" type="QString" value="" />
              <Option name="properties" type="Map" />
              <Option name="type" type="QString" value="collection" />
            </Option>
          </data_defined_properties>
        </layer>
      </symbol>
      <symbol type="fill" name="8" alpha="1" clip_to_extent="1" force_rhr="0" frame_rate="10" is_animated="0">
        <layer class="SimpleFill" enabled="1" locked="0" pass="0">
          <Option type="Map">
            <Option name="color" type="QString" value="69,90,100,38" />
            <Option name="style" type="QString" value="solid" />
            <Option name="outline_color" type="QString" value="69,90,100,255" />
            <Option name="outline_style" type="QString" value="solid" />
            <Option name="outline_width" type="QString" value="0.5" />
            <Option name="outline_width_unit" type="QString" value="MM" />
            <Option name="joinstyle" type="QString" value="round" />
          </Option>
          <data_defined_properties>
            <Option type="Map">
              <Option name="name" type="QString" value="" />
              <Option name="properties" type="Map" />
              <Option name="type" type="QString" value="collection" />
            </Option>
          </data_defined_properties>
        </layer>
      </symbol>
      <symbol type="fill" name="9" alpha="1" clip_to_extent="1" force_rhr="0" frame_rate="10" is_animated="0">
        <layer class="SimpleFill" enabled="1" locked="0" pass="0">
          <Option type="Map">
            <Option name="color" type="QString" value="69,90,100,38" />
            <Option name="style" type="QString" value="solid" />
            <Option name="outline_color" type="QString" value="69,90,100,255" />
            <Option name="outline_style" type="QString" value="dash" />
            <Option name="outline_width" type="QString" value="0.5" />
            <Option name="outline_width_unit" type="QString" value="MM" />
            <Option name="joinstyle" type="QString" value="round" />
          </Option>
          <data_defined_properties>
            <Option type="Map">
              <Option name="name" type="QString" value="" />
              <Option name="properties" type="Map" />
              <Option name="type" type="QString" value="collection" />
            </Option>
          </data_defined_properties>
        </layer>
      </symbol>
    </symbols>
  </renderer-v2>
  <labeling type="rule-based">
    <rules key="{labels-root}">
      <rule key="{label-0}" description="Station name" scalemaxdenom="70000" scalemindenom="1">
        <settings calloutType="simple">
          <text-style fieldName="&quot;name&quot;" isExpression="1" fontFamily="Sans Serif" fontSize="7.0" fontSizeUnit="Point" fontWeight="50" textColor="33,33,33,255" textOpacity="1" fontItalic="0" namedStyle="Regular" multilineHeight="1" allowHtml="0">
            <text-buffer bufferDraw="1" bufferSize="0.8" bufferSizeUnits="MM" bufferColor="255,255,255,230" bufferOpacity="0.9" bufferJoinStyle="128" bufferNoFill="1" />
          </text-style>
          <text-format wrapChar="" autoWrapLength="0" multilineAlign="0" plussign="0" reverseDirectionSymbol="0" />
          <placement placement="8" placementFlags="10" dist="1.2" distUnits="MM" priority="5" repeatDistance="0" overrunDistance="0" offsetType="0" quadOffset="4" centroidInside="1" fitInPolygonOnly="0" />
          <rendering obstacle="1" obstacleFactor="1" obstacleType="1" upsidedownLabels="0" displayAll="0" mergeLines="0" labelPerPart="0" limitNumLabels="0" maxNumLabels="2000" fontLimitPixelSize="0" scaleVisibility="0" drawLabels="1" />
        </settings>
      </rule>
    </rules>
  </labeling>
  <layerGeometryType>2</layerGeometryType>
</qgis>
